CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    price DECIMAL(10, 2) NOT NULL
);

-- Seed 25 products across 5 categories (5 per category)

-- Electronics
INSERT INTO products (name, description, category, price) VALUES
('Wireless Headphones', 'Noise-canceling over-ear headphones with 30-hour battery life and Bluetooth 5.0 connectivity. Features adaptive noise cancellation that adjusts to your environment and premium memory foam ear cushions for all-day comfort.', 'electronics', 79.99),
('Smart Watch Pro', 'Advanced fitness tracker with heart rate monitoring, GPS tracking, sleep analysis, and blood oxygen measurement. Water resistant to 50 meters with a bright AMOLED display visible in direct sunlight.', 'electronics', 199.99),
('Portable Charger', 'Ultra-compact 20000mAh power bank with dual USB-C ports and fast charging support. Charges a smartphone up to five times on a single charge with intelligent power management circuitry.', 'electronics', 34.99),
('Bluetooth Speaker', 'Waterproof portable speaker with 360-degree sound and 24-hour battery life. Features a rugged design that survives drops, dust, and submersion in water up to one meter deep.', 'electronics', 49.99),
('Wireless Earbuds', 'True wireless earbuds with active noise cancellation, transparency mode, and spatial audio support. The compact charging case provides an additional 24 hours of listening time.', 'electronics', 129.99);

-- Clothing
INSERT INTO products (name, description, category, price) VALUES
('Merino Wool Sweater', 'Lightweight merino wool sweater that naturally regulates temperature and resists odors. Machine washable with a classic crew neck design suitable for layering in any season.', 'clothing', 89.99),
('Waterproof Hiking Jacket', 'Breathable three-layer waterproof jacket designed for mountain hiking and trail running. Features sealed seams, adjustable hood, and multiple ventilation zippers for temperature control.', 'clothing', 159.99),
('Running Shoes Ultra', 'Lightweight running shoes with responsive foam cushioning and a breathable mesh upper. The carbon fiber plate provides energy return for faster times on road and track surfaces.', 'clothing', 149.99),
('Down Insulated Vest', 'Packable goose down vest rated to minus 20 degrees with water-resistant outer shell. Weighs only 200 grams and compresses into its own pocket for easy travel and storage.', 'clothing', 119.99),
('Stretch Denim Jeans', 'Comfortable stretch denim jeans with a modern slim fit and sustainable organic cotton blend. Features reinforced stitching and a hidden security pocket for valuables while traveling.', 'clothing', 69.99);

-- Home
INSERT INTO products (name, description, category, price) VALUES
('Robot Vacuum Cleaner', 'Intelligent robot vacuum with laser navigation, mopping capability, and automatic dustbin emptying. Maps your entire home and creates efficient cleaning paths for each room.', 'home', 399.99),
('Air Purifier HEPA', 'Medical-grade HEPA air purifier that captures 99.97 percent of airborne particles including dust, pollen, pet dander, and smoke. Covers rooms up to 500 square feet with whisper-quiet operation.', 'home', 249.99),
('Smart Thermostat', 'Learning thermostat that automatically adjusts temperature based on your habits and preferences. Compatible with all major HVAC systems and controllable via smartphone app or voice assistant.', 'home', 179.99),
('Ceramic Cookware Set', 'Eight-piece ceramic non-stick cookware set free from PFAS and PTFE chemicals. Even heat distribution with stainless steel handles that stay cool on the stovetop.', 'home', 129.99),
('Memory Foam Pillow', 'Ergonomic memory foam pillow with cooling gel layer and adjustable loft. Contoured design supports natural neck alignment for side, back, and stomach sleepers.', 'home', 59.99);

-- Books
INSERT INTO products (name, description, category, price) VALUES
('Deep Learning Fundamentals', 'Comprehensive guide to neural network architectures covering convolutional networks, transformers, and generative models with practical Python code examples and mathematical foundations.', 'books', 49.99),
('The Data Pipeline Handbook', 'Practical guide to building modern data pipelines using streaming technologies like Kafka, Pathway, and Apache Flink. Covers real-time ETL, change data capture, and event-driven architectures.', 'books', 44.99),
('Vector Search in Practice', 'Hands-on guide to implementing similarity search systems using vector databases like Milvus, including embedding strategies, index selection, and hybrid search techniques.', 'books', 39.99),
('Sustainable Gardening', 'Complete guide to organic gardening techniques including composting, companion planting, natural pest control, and water conservation strategies for home gardens.', 'books', 24.99),
('Modern Architecture Design', 'Explores contemporary architectural movements and sustainable building practices featuring stunning photography and detailed analysis of award-winning structures worldwide.', 'books', 54.99);

-- Sports
INSERT INTO products (name, description, category, price) VALUES
('Carbon Fiber Tennis Racket', 'Professional-grade tennis racket with carbon fiber frame providing excellent power and control. Features a vibration dampening system and optimized string pattern for spin generation.', 'sports', 189.99),
('Yoga Mat Premium', 'Extra-thick non-slip yoga mat made from natural rubber with alignment markings. Provides superior cushioning for joints while maintaining stability during balance poses.', 'sports', 34.99),
('Adjustable Dumbbell Set', 'Space-saving adjustable dumbbells that replace 15 pairs of weights. Quick-change mechanism allows weight adjustment from 5 to 52.5 pounds in 2.5 pound increments.', 'sports', 299.99),
('Cycling Computer GPS', 'Advanced cycling computer with GPS navigation, power meter compatibility, and performance analytics. Displays speed, cadence, heart rate, and elevation with a sunlight-readable color screen.', 'sports', 249.99),
('Insulated Water Bottle', 'Double-wall vacuum insulated stainless steel water bottle that keeps drinks cold for 24 hours or hot for 12 hours. BPA-free with a leak-proof lid and wide mouth for easy cleaning.', 'sports', 29.99);
