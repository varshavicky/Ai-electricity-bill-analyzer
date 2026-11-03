-- SQLite schema (app.db used by Flask)

CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(190) NOT NULL UNIQUE,
    password_hash VARCHAR(190) NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS electricity_bill (
    customer_id INT NOT NULL AUTO_INCREMENT,
    tv REAL NOT NULL,
    fan REAL NOT NULL,
    ac REAL NOT NULL,
    refrigerator REAL NOT NULL,
    washing_machine REAL NOT NULL,
    geyser REAL NOT NULL,
    lights REAL NOT NULL,
    units_consumed REAL NOT NULL,
    bill_amount REAL NOT NULL,
    PRIMARY KEY (customer_id)
);

