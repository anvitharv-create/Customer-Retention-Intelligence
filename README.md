# Supply Chain and Inventory Analysis Using Big Data Analytics

An end-to-end supply chain analytics project using **Python, PySpark, Apache Spark, Spark SQL, Parquet, and Streamlit** to analyze product sales, warehouse performance, supplier activity, regional performance, and inventory reorder requirements through an interactive web dashboard.

**Live Dashboard:** [Open the Streamlit App](https://supply-chain-big-data-o9gobhu445vgvrbn2wvou5.streamlit.app)

**GitHub Repository:** [Supply-Chain-Big-Data](https://github.com/anvitharv-create/Supply-Chain-Big-Data)

---

## Business Problem

Supply chain operations generate data about product sales, inventory levels, warehouses, suppliers, costs, prices, and demand forecasts. Analyzing these records manually can make it difficult to identify business patterns and inventory requirements.

This project addresses that problem by building a data processing and analytics pipeline that transforms raw supply chain data into meaningful insights. It helps users compare product revenue, warehouse performance, supplier lead times, regional sales, and inventory records that meet predefined reorder conditions.

The results are presented through an interactive dashboard, allowing users to explore the data using filters, KPI cards, and visualizations.

## Key Insights

The following results were obtained from the project's dataset and analytical pipeline:

- **Top product by revenue:** SKU_38, generating ₹1,057,641.51.
- **Top warehouse by revenue:** WH_2, generating ₹7,395,653.40.
- **Highest supplier order quantity:** SUP_7, with 239,283 units.
- **Top region by revenue:** North, generating ₹8,411,171.01.
- **Records requiring reorder:** 5,041.
- **Records with sufficient stock:** 86,209.
- **Data validation:** All 91,250 records passed the defined validation checks, with zero missing values, duplicate records, or invalid records.

**Business takeaway:** The results help users identify high-revenue products, compare warehouse and regional performance, examine supplier lead times, and prioritize inventory records that meet the defined reorder condition.

*Note: These results describe the supplied dataset. The Stockout_Flag column contains no positive stockout records, so the project does not claim to have observed actual stockout events.*

## What This Project Delivers

### Data Ingestion
- Loads the raw CSV dataset using PySpark.
- Infers the data schema and converts the date field to an appropriate date type.
- Inspects the record count, columns, schema, and sample records.

### Data Validation
- Checks for missing values and duplicate records.
- Validates numerical fields and business rules.
- Verifies the quality of the dataset before further processing.

### Feature Engineering
Creates additional analytical fields:

- `Revenue`
- `Total_Cost`
- `Profit`
- `Inventory_Value`
- `Demand_Variance`
- `Reorder_Status`
- `Stock_Level_Category`

For example:

- Revenue = Units Sold × Unit Price
- Total Cost = Units Sold × Unit Cost
- Profit = Revenue − Total Cost
- Inventory Value = Inventory Level × Unit Cost
- Demand Variance = Units Sold − Demand Forecast

### Parquet Storage
- Stores the processed dataset in Parquet format.
- Keeps the raw dataset separate from processed data.
- Allows the processed data to be reused for analysis.

### Product Analysis
- Compares products using units sold, revenue, profit, and average inventory.
- Identifies products with the highest revenue in the analyzed results.

### Warehouse Analysis
- Compares warehouses by units sold, revenue, and inventory-related measures.
- Helps users understand differences in warehouse performance.

### Supplier Analysis
- Analyzes supplier lead times, order quantities, costs, and profit.
- Supports comparisons between suppliers using the available metrics.

### Regional Analysis
- Compares North, South, East, and West.
- Examines regional units sold, revenue, profit, and average inventory.

### Reorder Analysis
- Identifies records where inventory is less than or equal to the reorder point.
- Groups records into `REORDER_REQUIRED` and `STOCK_SUFFICIENT`.

### Spark SQL
- Executes SQL-based analytical queries on the processed Spark DataFrame.
- Provides separate queries for product, warehouse, supplier, and regional analysis.

### Interactive Dashboard
- Displays KPI cards and interactive Plotly charts.
- Provides dedicated analysis pages.
- Supports filtering by region, product, warehouse, supplier, and time period.
- Includes custom-day filtering and a Data Explorer for inspecting filtered records.

## Dashboard

The deployed Streamlit dashboard contains the following pages:

1. **Dashboard** — Overall supply chain metrics.
2. **Regional Analysis** — Regional sales, revenue, profit, and inventory.
3. **Product Analysis** — Product-level performance.
4. **Warehouse Analysis** — Warehouse-level performance and inventory measures.
5. **Supplier Analysis** — Supplier lead times, order quantities, and profitability.
6. **Reorder Analysis** — Inventory reorder status and related measures.
7. **Data Explorer** — Filtered records and data exploration.

**Try the dashboard:** [Supply Chain Analytics — Live Demo](https://supply-chain-big-data-o9gobhu445vgvrbn2wvou5.streamlit.app)

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Apache Spark | Distributed data processing framework |
| PySpark | Python interface for Spark DataFrame processing |
| Spark SQL | SQL-based analytics |
| Parquet | Processed-data storage |
| Streamlit | Interactive web dashboard |
| Pandas | Supporting data manipulation |
| Plotly | Interactive charts and visualizations |
| Git | Version control |
| GitHub | Source code repository and project management |
| VS Code | Development environment |

## Dataset

- **Dataset:** High-Dimensional Supply Chain Inventory Dataset
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/ziya07/high-dimensional-supply-chain-inventory-dataset)
- **Original records:** 91,250
- **Original columns:** 15
- **Processed columns:** 22
- **Missing values:** 0
- **Duplicate records:** 0
- **Invalid records under defined validation rules:** 0

### Key Dataset Fields

| Field | Description |
|---|---|
| `Date` | Date associated with the record |
| `SKU_ID` | Product identifier |
| `Warehouse_ID` | Warehouse identifier |
| `Supplier_ID` | Supplier identifier |
| `Region` | Region associated with the record |
| `Units_Sold` | Number of units sold |
| `Inventory_Level` | Available inventory quantity |
| `Supplier_Lead_Time_Days` | Supplier lead time in days |
| `Reorder_Point` | Inventory threshold for reordering |
| `Order_Quantity` | Quantity ordered |
| `Unit_Cost` | Cost per unit |
| `Unit_Price` | Selling price per unit |
| `Promotion_Flag` | Promotion indicator |
| `Stockout_Flag` | Stockout indicator |
| `Demand_Forecast` | Forecasted demand |

**Dataset limitation:** The dataset contains approximately 91,250 records. It demonstrates a Spark-based analytics workflow, but it should not be described as extremely large industrial Big Data. The same processing approach can be extended to larger datasets.

## Project Workflow

```text
Raw CSV Dataset
       |
       v
Data Ingestion using PySpark
       |
       v
Data Validation
       |
       v
Feature Engineering
       |
       v
Processed Data in Parquet
       |
       v
PySpark Analytics
       |
       v
Spark SQL Analysis
       |
       v
Product, Warehouse, Supplier,
Regional and Reorder Analysis
       |
       v
Interactive Streamlit Dashboard
       |
       v
GitHub Repository and Deployment
```

### Why PySpark?

PySpark provides DataFrame operations and transformations for processing structured data. It allows the project to demonstrate a Spark-based processing pipeline that can be extended to larger datasets.

### Why Parquet?

Parquet is a columnar storage format suited to analytical workloads. It allows the project to store processed data separately from the raw CSV and reuse the processed dataset for subsequent analysis.

## Sample Spark SQL Queries

The project registers the processed DataFrame as a temporary view named `supply_chain` within the Spark SQL analysis session.

### 1. Records Requiring Reorder

```sql
SELECT
    SKU_ID,
    Warehouse_ID,
    Inventory_Level,
    Reorder_Point,
    Reorder_Status
FROM supply_chain
WHERE Inventory_Level <= Reorder_Point
ORDER BY Inventory_Level ASC;
```

This query identifies records where the inventory level has reached or fallen below the defined reorder point.

### 2. Supplier Lead-Time Analysis

```sql
SELECT
    Supplier_ID,
    AVG(Supplier_Lead_Time_Days) AS avg_lead_time,
    SUM(Order_Quantity) AS total_order_quantity,
    SUM(Total_Cost) AS total_cost,
    SUM(Profit) AS total_profit
FROM supply_chain
GROUP BY Supplier_ID
ORDER BY avg_lead_time DESC;
```

This query compares suppliers using average lead time, order quantity, total cost, and profit.

### 3. Regional Revenue Analysis

```sql
SELECT
    Region,
    SUM(Units_Sold) AS total_units_sold,
    SUM(Revenue) AS total_revenue,
    SUM(Profit) AS total_profit,
    AVG(Inventory_Level) AS avg_inventory
FROM supply_chain
GROUP BY Region
ORDER BY total_revenue DESC;
```

This query compares regional sales and profitability using the processed dataset.

## Project Structure

```text
Supply-Chain-Big-Data/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── supply_chain_dataset1.csv
│   ├── processed/
│   │   └── supply_chain_processed/
│   └── dashboard/
├── docs/
├── notebooks/
├── output/
├── sql/
│   ├── product_analysis.sql
│   ├── warehouse_analysis.sql
│   ├── supplier_analysis.sql
│   └── regional_analysis.sql
├── src/
│   ├── spark_test.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── analytics.py
│   ├── sql_analysis.py
│   └── export_dashboard_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The raw CSV and processed Parquet data must be available in their expected locations for the corresponding scripts to run. Generated data files may be excluded from Git depending on the repository configuration.*

## How to Run Locally

### Prerequisites

- Python 3.12
- Java installed and configured through `JAVA_HOME`
- Git
- The project dataset in the expected raw-data directory

The development environment for this project used Python 3.12.8, Java 25.0.2, and PySpark 4.2.0.

### 1. Clone the Repository

```bash
git clone https://github.com/anvitharv-create/Supply-Chain-Big-Data.git
cd Supply-Chain-Big-Data
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure Java is installed and `JAVA_HOME` is configured before running PySpark.

### 4. Test the Spark Installation

```powershell
python .\src\spark_test.py
```

### 5. Run the Data Processing Pipeline

Run the following scripts from the project root, in order:

```powershell
python .\src\data_ingestion.py
python .\src\data_validation.py
python .\src\feature_engineering.py
python .\src\analytics.py
```

The feature-engineering step generates the processed Parquet dataset under:

```text
data/processed/supply_chain_processed
```

### 6. Run Spark SQL Analysis

For example:

```powershell
python .\src\sql_analysis.py .\sql\product_analysis.sql
python .\src\sql_analysis.py .\sql\warehouse_analysis.sql
python .\src\sql_analysis.py .\sql\supplier_analysis.sql
python .\src\sql_analysis.py .\sql\regional_analysis.sql
```

### 7. Launch the Dashboard

```powershell
streamlit run .\dashboard\app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to explore the dashboard.

## Results and Limitations

### Results

- Processed 91,250 source records through the defined validation workflow.
- Created a feature-engineered dataset containing 22 columns.
- Stored the processed data in Parquet format.
- Implemented PySpark and Spark SQL analysis.
- Developed an interactive dashboard with dedicated analytical pages.
- Maintained the source code in GitHub.
- Deployed the Streamlit dashboard as a web application.

### Limitations

- The dataset contains approximately 91,250 records and is not an extremely large industrial dataset.
- The `Stockout_Flag` column contains no positive stockout records, preventing analysis of observed stockout events.
- The current system primarily performs descriptive analytics and visualization; it does not implement a trained demand-forecasting model.
- Reorder status is determined using the defined relationship between `Inventory_Level` and `Reorder_Point`.
- The project does not establish supplier defect rates or order fulfillment reliability because the available data does not support those measurements.
- The dashboard is an analytical application rather than a complete enterprise inventory management system.

## Future Improvements

Potential extensions include:

- Demand forecasting using machine learning.
- Automated reorder alerts and replenishment recommendations.
- Inventory optimization based on demand and supplier lead times.
- Real-time data ingestion using streaming technologies such as Apache Kafka.
- Cloud-based Spark processing for larger datasets.
- Integration with live inventory databases.
- Additional data-quality tests and automated reporting.

## Author

**Anvitha R V**

GitHub: [anvitharv-create](https://github.com/anvitharv-create)

---

*This project demonstrates an end-to-end analytics workflow combining data ingestion, validation, feature engineering, Parquet storage, PySpark processing, Spark SQL analysis, interactive visualization, version control, and web deployment.*


## Dashboard Screenshots

### 1. Analytics Overview
![Analytics Overview](screenshots/analytics-overview.png)

### 2. High-Risk Customers
![High-Risk Customers](screenshots/high-risk-customers.png)

### 3. Predict Churn
![Predict Churn](screenshots/predict-churn.png)

### 4. Model Performance
![Model Performance](screenshots/model-performance.png)
