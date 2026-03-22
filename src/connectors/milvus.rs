// Copyright © 2026 Pathway

use std::collections::{HashMap, HashSet};
use std::mem::take;

use log::info;
use milvus::client::Client as MilvusClient;
use milvus::data::FieldColumn;
use milvus::index::{IndexParams, IndexType, MetricType};
use milvus::mutate::{DeleteOptions, InsertOptions};
use milvus::schema::{CollectionSchemaBuilder, FieldSchema};
use milvus::value::ValueVec;
use tokio::runtime::Runtime as TokioRuntime;

use crate::async_runtime::create_async_tokio_runtime;
use crate::connectors::data_format::FormatterContext;
use crate::connectors::{WriteError, Writer};
use crate::engine::Value;

/// Configuration for a single vector column in Milvus.
#[derive(Debug, Clone)]
pub struct VectorColumnConfig {
    pub milvus_type: MilvusFieldType,
    pub dimension: Option<i64>,
    pub index_type: String,
    pub metric_type: String,
}

/// Supported Milvus field types for vector columns.
#[derive(Debug, Clone, Copy)]
pub enum MilvusFieldType {
    FloatVector,
    BinaryVector,
}

/// Primary key type used to track deletes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MilvusPrimaryKey {
    Int64(i64),
    VarChar(String),
}

/// A buffered row ready for upsert into Milvus.
struct BufferedRow {
    pk: MilvusPrimaryKey,
    field_values: HashMap<String, FieldValue>,
}

/// Internal representation of a field value before conversion to Milvus FieldColumn.
enum FieldValue {
    Bool(bool),
    Int64(i64),
    Float(f32),
    Double(f64),
    String(String),
    FloatVector(Vec<f32>),
    BinaryVector(Vec<u8>),
    Json(Vec<u8>),
}

pub struct MilvusWriter {
    runtime: TokioRuntime,
    client: MilvusClient,
    collection_name: String,
    primary_key_column: String,
    vector_columns: HashMap<String, VectorColumnConfig>,
    value_field_names: Vec<String>,
    max_batch_size: Option<usize>,
    create_collection_if_missing: bool,

    upsert_buffer: Vec<BufferedRow>,
    delete_buffer: Vec<MilvusPrimaryKey>,
    collection_initialized: bool,
}

impl MilvusWriter {
    pub fn new(
        uri: String,
        collection_name: String,
        primary_key_column: String,
        vector_columns: HashMap<String, VectorColumnConfig>,
        value_field_names: Vec<String>,
        max_batch_size: Option<usize>,
        create_collection_if_missing: bool,
        token: Option<String>,
    ) -> Result<Self, WriteError> {
        let runtime = create_async_tokio_runtime()?;

        let client = runtime
            .block_on(async {
                if let Some(ref token) = token {
                    // Token format: "username:password" or raw token
                    let parts: Vec<&str> = token.splitn(2, ':').collect();
                    if parts.len() == 2 {
                        milvus::client::ClientBuilder::new(uri)
                            .username(parts[0])
                            .password(parts[1])
                            .build()
                            .await
                    } else {
                        MilvusClient::new(uri).await
                    }
                } else {
                    MilvusClient::new(uri).await
                }
            })
            .map_err(|e| WriteError::Milvus(format!("failed to connect to Milvus: {e}")))?;

        Ok(MilvusWriter {
            runtime,
            client,
            collection_name,
            primary_key_column,
            vector_columns,
            value_field_names,
            max_batch_size,
            create_collection_if_missing,
            upsert_buffer: Vec::new(),
            delete_buffer: Vec::new(),
            collection_initialized: false,
        })
    }

    fn ensure_collection(&mut self) -> Result<(), WriteError> {
        if self.collection_initialized {
            return Ok(());
        }

        let exists = self
            .runtime
            .block_on(self.client.has_collection(&self.collection_name))
            .map_err(|e| WriteError::Milvus(format!("has_collection failed: {e}")))?;

        if !exists {
            if !self.create_collection_if_missing {
                return Err(WriteError::Milvus(format!(
                    "collection '{}' does not exist and create_collection_if_missing is false",
                    self.collection_name
                )));
            }
            self.create_collection()?;
        }

        self.collection_initialized = true;
        Ok(())
    }

    fn create_collection(&mut self) -> Result<(), WriteError> {
        let collection_name = self.collection_name.clone();
        let pk_column = self.primary_key_column.clone();
        let vector_columns = self.vector_columns.clone();

        self.runtime.block_on(async {
            let mut schema_builder =
                CollectionSchemaBuilder::new(&collection_name, "Created by Pathway");

            // Add primary key field
            schema_builder.add_field(FieldSchema::new_primary_int64(&pk_column, "", false));

            // Add vector fields
            for (name, config) in &vector_columns {
                match config.milvus_type {
                    MilvusFieldType::FloatVector => {
                        let dim = config.dimension.unwrap_or(128);
                        schema_builder.add_field(FieldSchema::new_float_vector(name, "", dim));
                    }
                    MilvusFieldType::BinaryVector => {
                        let dim = config.dimension.unwrap_or(128);
                        schema_builder.add_field(FieldSchema::new_binary_vector(name, "", dim));
                    }
                }
            }

            // Enable dynamic fields for non-vector, non-pk columns
            schema_builder.enable_dynamic_field();

            let schema = schema_builder
                .build()
                .map_err(|e| WriteError::Milvus(format!("schema build failed: {e}")))?;

            self.client
                .create_collection(schema, None)
                .await
                .map_err(|e| WriteError::Milvus(format!("create_collection failed: {e}")))?;

            // Create indexes on vector fields
            for (name, config) in &vector_columns {
                let index_type = match config.index_type.as_str() {
                    "FLAT" => IndexType::Flat,
                    "IVF_FLAT" => IndexType::IvfFlat,
                    "IVF_SQ8" => IndexType::IvfSQ8,
                    "IVF_PQ" => IndexType::IvfPQ,
                    "HNSW" => IndexType::HNSW,
                    _ => IndexType::Flat, // AUTOINDEX fallback
                };

                let metric_type = match config.metric_type.as_str() {
                    "L2" => MetricType::L2,
                    "IP" => MetricType::IP,
                    "HAMMING" => MetricType::HAMMING,
                    "JACCARD" => MetricType::JACCARD,
                    _ => MetricType::IP,
                };

                let index_params = IndexParams::new(
                    name.clone(),
                    index_type,
                    metric_type,
                    HashMap::new(),
                );

                self.client
                    .create_index(&collection_name, name, index_params)
                    .await
                    .map_err(|e| WriteError::Milvus(format!("create_index failed: {e}")))?;
            }

            // Load the collection into memory
            self.client
                .load_collection(&collection_name, None)
                .await
                .map_err(|e| WriteError::Milvus(format!("load_collection failed: {e}")))?;

            info!("Created and loaded Milvus collection '{collection_name}'");
            Ok(())
        })
    }

    fn extract_pk(&self, values: &[Value]) -> Option<MilvusPrimaryKey> {
        let pk_idx = self
            .value_field_names
            .iter()
            .position(|n| n == &self.primary_key_column)?;
        match values.get(pk_idx)? {
            Value::Int(i) => Some(MilvusPrimaryKey::Int64(*i)),
            Value::String(s) => Some(MilvusPrimaryKey::VarChar(s.to_string())),
            _ => None,
        }
    }

    fn extract_row(&self, values: &[Value]) -> HashMap<String, FieldValue> {
        let mut fields = HashMap::new();
        for (i, name) in self.value_field_names.iter().enumerate() {
            if let Some(val) = values.get(i) {
                let fv = match val {
                    Value::Bool(b) => Some(FieldValue::Bool(*b)),
                    Value::Int(n) => Some(FieldValue::Int64(*n)),
                    Value::Float(f) => Some(FieldValue::Double(f.into_inner())),
                    Value::String(s) => Some(FieldValue::String(s.to_string())),
                    Value::Bytes(b) => Some(FieldValue::BinaryVector(b.to_vec())),
                    Value::Json(j) => {
                        let json_bytes = serde_json::to_vec(&**j)
                            .unwrap_or_default();
                        Some(FieldValue::Json(json_bytes))
                    }
                    Value::FloatArray(arr) => {
                        let floats: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
                        Some(FieldValue::FloatVector(floats))
                    }
                    Value::Tuple(elems) => {
                        // Treat tuples of floats as float vectors
                        let mut floats = Vec::with_capacity(elems.len());
                        for elem in elems.iter() {
                            match elem {
                                Value::Float(f) => floats.push(f.into_inner() as f32),
                                Value::Int(i) => floats.push(*i as f32),
                                _ => return fields, // bail on non-numeric tuples
                            }
                        }
                        Some(FieldValue::FloatVector(floats))
                    }
                    _ => None,
                };
                if let Some(fv) = fv {
                    fields.insert(name.clone(), fv);
                }
            }
        }
        fields
    }

    fn flush_upserts(&mut self) -> Result<(), WriteError> {
        if self.upsert_buffer.is_empty() {
            return Ok(());
        }

        let rows = take(&mut self.upsert_buffer);
        let collection_name = self.collection_name.clone();

        // Build columnar FieldColumns from buffered rows
        let field_columns = self.build_field_columns(&rows)?;

        self.runtime.block_on(async {
            self.client
                .upsert(&collection_name, field_columns, Some(InsertOptions::default()))
                .await
                .map_err(|e| WriteError::Milvus(format!("upsert failed: {e}")))?;
            Ok(())
        })
    }

    fn flush_deletes(&mut self) -> Result<(), WriteError> {
        if self.delete_buffer.is_empty() {
            return Ok(());
        }

        let pks = take(&mut self.delete_buffer);
        let collection_name = self.collection_name.clone();

        // Build delete expression based on PK type
        let ids = match &pks[0] {
            MilvusPrimaryKey::Int64(_) => {
                let vals: Vec<i64> = pks
                    .into_iter()
                    .filter_map(|pk| match pk {
                        MilvusPrimaryKey::Int64(i) => Some(i),
                        _ => None,
                    })
                    .collect();
                ValueVec::Long(vals)
            }
            MilvusPrimaryKey::VarChar(_) => {
                let vals: Vec<String> = pks
                    .into_iter()
                    .filter_map(|pk| match pk {
                        MilvusPrimaryKey::VarChar(s) => Some(s),
                        _ => None,
                    })
                    .collect();
                ValueVec::String(vals)
            }
        };

        let options = DeleteOptions::with_ids(ids);

        self.runtime.block_on(async {
            self.client
                .delete(&collection_name, &options)
                .await
                .map_err(|e| WriteError::Milvus(format!("delete failed: {e}")))?;
            Ok(())
        })
    }

    fn build_field_columns(&self, rows: &[BufferedRow]) -> Result<Vec<FieldColumn>, WriteError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let mut columns: HashMap<String, Vec<&FieldValue>> = HashMap::new();

        // Collect all field names from all rows
        for row in rows {
            for (name, val) in &row.field_values {
                columns.entry(name.clone()).or_default().push(val);
            }
        }

        let mut field_columns = Vec::new();

        for (name, values) in &columns {
            let is_vector = self.vector_columns.contains_key(name);
            let is_pk = name == &self.primary_key_column;

            let fc = if is_pk {
                // Primary key column - Int64
                let mut longs = Vec::with_capacity(values.len());
                for val in values {
                    match val {
                        FieldValue::Int64(i) => longs.push(*i),
                        FieldValue::String(_) => {
                            // VarChar PK - handle separately
                            let strings: Vec<String> = values
                                .iter()
                                .filter_map(|v| match v {
                                    FieldValue::String(s) => Some(s.clone()),
                                    _ => None,
                                })
                                .collect();
                            let schema = FieldSchema::new_primary_varchar(name, "", false, 512);
                            return Ok(vec![FieldColumn::new(&schema, strings)]);
                        }
                        _ => {
                            return Err(WriteError::Milvus(format!(
                                "primary key '{name}' has unsupported type"
                            )));
                        }
                    }
                }
                let schema = FieldSchema::new_primary_int64(name, "", false);
                FieldColumn::new(&schema, longs)
            } else if is_vector {
                let config = &self.vector_columns[name];
                match config.milvus_type {
                    MilvusFieldType::FloatVector => {
                        let dim = config.dimension.unwrap_or(128);
                        let mut all_floats = Vec::new();
                        for val in values {
                            match val {
                                FieldValue::FloatVector(v) => all_floats.extend_from_slice(v),
                                _ => {
                                    return Err(WriteError::Milvus(format!(
                                        "vector field '{name}' has non-float-vector value"
                                    )));
                                }
                            }
                        }
                        let schema = FieldSchema::new_float_vector(name, "", dim);
                        FieldColumn::new(&schema, all_floats)
                    }
                    MilvusFieldType::BinaryVector => {
                        let dim = config.dimension.unwrap_or(128);
                        let mut all_bytes = Vec::new();
                        for val in values {
                            match val {
                                FieldValue::BinaryVector(v) => all_bytes.extend_from_slice(v),
                                _ => {
                                    return Err(WriteError::Milvus(format!(
                                        "binary vector field '{name}' has non-binary value"
                                    )));
                                }
                            }
                        }
                        let schema = FieldSchema::new_binary_vector(name, "", dim);
                        FieldColumn::new(&schema, all_bytes)
                    }
                }
            } else {
                // Dynamic/scalar field - convert based on first value type
                match values[0] {
                    FieldValue::Bool(_) => {
                        let bools: Vec<bool> = values
                            .iter()
                            .filter_map(|v| match v {
                                FieldValue::Bool(b) => Some(*b),
                                _ => None,
                            })
                            .collect();
                        let schema = FieldSchema::new_bool(name, "");
                        FieldColumn::new(&schema, bools)
                    }
                    FieldValue::Int64(_) => {
                        let longs: Vec<i64> = values
                            .iter()
                            .filter_map(|v| match v {
                                FieldValue::Int64(i) => Some(*i),
                                _ => None,
                            })
                            .collect();
                        let schema = FieldSchema::new_int64(name, "");
                        FieldColumn::new(&schema, longs)
                    }
                    FieldValue::Double(_) => {
                        let doubles: Vec<f64> = values
                            .iter()
                            .filter_map(|v| match v {
                                FieldValue::Double(d) => Some(*d),
                                _ => None,
                            })
                            .collect();
                        let schema = FieldSchema::new_double(name, "");
                        FieldColumn::new(&schema, doubles)
                    }
                    FieldValue::String(_) => {
                        let strings: Vec<String> = values
                            .iter()
                            .filter_map(|v| match v {
                                FieldValue::String(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect();
                        let schema = FieldSchema::new_varchar(name, "", 65535);
                        FieldColumn::new(&schema, strings)
                    }
                    FieldValue::Json(_) => {
                        let jsons: Vec<Vec<u8>> = values
                            .iter()
                            .filter_map(|v| match v {
                                FieldValue::Json(j) => Some(j.clone()),
                                _ => None,
                            })
                            .collect();
                        FieldColumn {
                            name: name.clone(),
                            dtype: milvus::proto::schema::DataType::Json,
                            value: ValueVec::Json(jsons),
                            dim: 1,
                            max_length: 0,
                            is_dynamic: false,
                        }
                    }
                    _ => continue,
                }
            };

            field_columns.push(fc);
        }

        Ok(field_columns)
    }
}

impl Writer for MilvusWriter {
    fn write(&mut self, data: FormatterContext) -> Result<(), WriteError> {
        self.ensure_collection()?;

        let pk = self
            .extract_pk(&data.values)
            .ok_or_else(|| WriteError::Milvus("could not extract primary key".into()))?;

        if data.diff > 0 {
            // Insertion/update
            let field_values = self.extract_row(&data.values);
            self.upsert_buffer.push(BufferedRow {
                pk,
                field_values,
            });

            if let Some(max_batch_size) = self.max_batch_size {
                if self.upsert_buffer.len() >= max_batch_size {
                    self.flush(true)?;
                }
            }
        } else {
            // Deletion
            self.delete_buffer.push(pk);
        }

        Ok(())
    }

    fn flush(&mut self, _forced: bool) -> Result<(), WriteError> {
        if self.upsert_buffer.is_empty() && self.delete_buffer.is_empty() {
            return Ok(());
        }

        // CDC-aware logic: filter out deletes for PKs that are also being upserted
        // (those represent updates, not true deletes)
        if !self.delete_buffer.is_empty() && !self.upsert_buffer.is_empty() {
            let upsert_pks: HashSet<&MilvusPrimaryKey> =
                self.upsert_buffer.iter().map(|r| &r.pk).collect();
            self.delete_buffer.retain(|pk| !upsert_pks.contains(pk));
        }

        // Flush upserts first, then remaining deletes
        self.flush_upserts()?;
        self.flush_deletes()?;

        Ok(())
    }

    fn name(&self) -> String {
        format!("Milvus({})", self.collection_name)
    }

    fn single_threaded(&self) -> bool {
        false
    }
}
