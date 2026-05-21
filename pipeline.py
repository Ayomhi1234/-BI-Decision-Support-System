# ============================================
# AI-POWERED BI DECISION SUPPORT SYSTEM
# AUTOMATED PIPELINE — RUNS ON NEW DATA
# ============================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import time
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from groq import Groq

# ============================================
# MAIN PIPELINE FUNCTION
# ============================================

def run_pipeline():
    print(f"\n{'='*50}")
    print(f" PIPELINE STARTED — {datetime.now().strftime('%B %d, %Y %H:%M')}")
    print(f"{'='*50}\n")

    # CREATE REPORT FOLDER WITH DATE STAMP
    report_name = f"BI_Report_{datetime.now().strftime('%B_%Y')}"
    report_folder = os.path.join("outputs", report_name)
    os.makedirs(report_folder, exist_ok=True)
    print(f"Report folder created: {report_folder}")

    # ============================================
    # STEP 1 — LOAD DATA
    # ============================================
    print("\nStep 1 — Loading data...")
    orders = pd.read_csv("olist_orders_dataset.csv")
    items = pd.read_csv("olist_order_items_dataset.csv")
    payment = pd.read_csv("olist_order_payments_dataset.csv")
    customer = pd.read_csv("olist_customers_dataset.csv")
    products = pd.read_csv("olist_products_dataset.csv")
    sellers = pd.read_csv("olist_sellers_dataset.csv")
    reviews = pd.read_csv("olist_order_reviews_dataset.csv")
    product_cat = pd.read_csv("product_category_name_translation.csv")
    print("All datasets loaded ✅")

    # ============================================
    # STEP 2 — CLEAN DATA
    # ============================================
    print("\nStep 2 — Cleaning data...")

    # Fix date columns in orders
    date_cols = ["order_purchase_timestamp", "order_approved_at",
                 "order_delivered_carrier_date", "order_delivered_customer_date",
                 "order_estimated_delivery_date"]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])

    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])

    # Drop missing approvals
    orders = orders.dropna(subset=["order_approved_at"])

    # Handle outliers in items
    Q1 = items["price"].quantile(0.25)
    Q3 = items["price"].quantile(0.75)
    IQR = Q3 - Q1
    items = items[(items["price"] >= Q1 - 1.5 * IQR) & (items["price"] <= Q3 + 1.5 * IQR)]

    # Handle outliers in payment
    Q1 = payment["payment_value"].quantile(0.25)
    Q3 = payment["payment_value"].quantile(0.75)
    IQR = Q3 - Q1
    payment = payment[(payment["payment_value"] >= Q1 - 1.5 * IQR) & (payment["payment_value"] <= Q3 + 1.5 * IQR)]
    payment = payment[~((payment["payment_type"] == "not_defined") & (payment["payment_value"] == 0))]

    # Fill missing values
    reviews["review_comment_title"] = reviews["review_comment_title"].fillna("No Title")
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("No Comment")

    # Translate product names — keep both columns for Power BI compatibility
    products = pd.merge(products, product_cat, on="product_category_name", how="outer")
    products["product_category_name_english"] = products["product_category_name_english"].fillna("Unknown")
    products["product_category_name"] = products["product_category_name_english"]

    # Drop duplicates
    orders = orders.drop_duplicates()
    items = items.drop_duplicates()
    payment = payment.drop_duplicates()

    print("Data cleaned ✅")

    # ============================================
    # STEP 3 — MERGE INTO MASTER DATAFRAME
    # ============================================
    print("\nStep 3 — Merging datasets...")
    merge = pd.merge(orders, items, on="order_id", how="inner")
    merge1 = pd.merge(merge, payment, on="order_id", how="inner")
    merge2 = pd.merge(merge1, customer, on="customer_id", how="inner")
    merge3 = pd.merge(merge2, products, on="product_id", how="inner")
    merge4 = pd.merge(merge3, sellers, on="seller_id", how="inner")
    master_df = pd.merge(merge4, reviews, on="order_id", how="left")
    print(f"Master dataframe created — {master_df.shape[0]} rows ✅")

    # ============================================
    # STEP 4 — FEATURE ENGINEERING
    # ============================================
    print("\nStep 4 — Feature engineering...")
    master_df["delivery_time"] = (master_df["order_delivered_customer_date"] - master_df["order_purchase_timestamp"]).dt.days
    master_df["total_order_value"] = master_df["price"] + master_df["freight_value"]
    master_df["order_month"] = master_df["order_purchase_timestamp"].dt.month
    master_df["order_year"] = master_df["order_purchase_timestamp"].dt.year
    print("Feature engineering done ✅")

    # ============================================
    # STEP 5 — RFM ANALYSIS
    # ============================================
    print("\nStep 5 — Running RFM analysis...")

    customer_orders = master_df.copy()
    if "customer_unique_id" not in customer_orders.columns:
        customer_orders = customer_orders.merge(
            customer[["customer_id", "customer_unique_id"]],
            on="customer_id", how="left"
        )

    rfm = customer_orders.groupby("customer_unique_id").agg(
        recency=("order_purchase_timestamp", "max"),
        frequency=("order_id", "count"),
        monetary=("price", "sum")
    ).reset_index()

    reference_date = rfm["recency"].max()
    rfm["recency"] = (reference_date - rfm["recency"]).dt.days

    rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])
    rfm["rfm_score"] = rfm["recency_score"].astype(int) + rfm["frequency_score"].astype(int) + rfm["monetary_score"].astype(int)

    def segment_customer(score):
        if score >= 12:
            return "Premium"
        elif score >= 9:
            return "Regular"
        elif score >= 6:
            return "Needs Attention"
        else:
            return "Churned"

    rfm["segment"] = rfm["rfm_score"].apply(segment_customer)
    rfm["churn_probability"] = (rfm["recency"] - rfm["recency"].min()) / (rfm["recency"].max() - rfm["recency"].min())
    print("RFM analysis done ✅")

    # ============================================
    # STEP 6 — LOAD SAVED ML MODELS
    # ============================================
    print("\nStep 6 — Loading ML models...")
    churn_model = joblib.load("churn_model.joblib")
    sales_model = joblib.load("sales_model.joblib")
    demand_model = joblib.load("demand_model.joblib")
    print("Models loaded ✅")

    # ============================================
    # STEP 7 — GENERATE VISUALIZATIONS
    # ============================================
    print("\nStep 7 — Generating visualizations...")

    # Chart 1 — Revenue Trends
    monthly_revenue = master_df.groupby("order_month")["price"].sum()
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_revenue.index.astype(str), monthly_revenue.values, marker="o")
    plt.title(f"Monthly Revenue Trends — {datetime.now().strftime('%B %Y')}")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(report_folder, "01_monthly_revenue.png"))
    plt.close()

    # Chart 2 — Top 10 Products
    top_products = master_df.groupby("product_category_name_english")["price"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_products.plot(kind="barh", color="green")
    plt.title("Top 10 Best Performing Products")
    plt.xlabel("Total Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(report_folder, "02_top_products.png"))
    plt.close()

    # Chart 3 — Worst 10 Products
    worst_products = master_df.groupby("product_category_name_english")["price"].sum().sort_values(ascending=True).head(10)
    plt.figure(figsize=(12, 6))
    worst_products.plot(kind="barh", color="red")
    plt.title("Top 10 Worst Performing Products")
    plt.xlabel("Total Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(report_folder, "03_worst_products.png"))
    plt.close()

    # Chart 4 — Regional Performance
    regional = master_df.groupby("customer_state")["price"].sum().sort_values(ascending=False)
    plt.figure(figsize=(14, 6))
    regional.plot(kind="bar")
    plt.title("Regional Performance by State")
    plt.xlabel("State")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_folder, "04_regional_performance.png"))
    plt.close()

    # Chart 5 — Customer Segments
    segment_counts = rfm["segment"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(segment_counts.values, labels=segment_counts.index, autopct="%1.1f%%")
    plt.title("Customer Segments")
    plt.tight_layout()
    plt.savefig(os.path.join(report_folder, "05_customer_segments.png"))
    plt.close()

    # Chart 6 — Payment Type Distribution
    payment_type = payment["payment_type"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(payment_type.values, labels=payment_type.index, autopct="%1.1f%%")
    plt.title("Payment Type Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(report_folder, "06_payment_types.png"))
    plt.close()

    print("All visualizations saved ✅")

    # ============================================
    # STEP 8 — GENERATE AI INSIGHTS
    # ============================================
    print("\nStep 8 — Generating AI insights...")

    total_revenue = master_df["price"].sum()
    avg_delivery = master_df["delivery_time"].mean()
    top_state = master_df.groupby("customer_state")["price"].sum().idxmax()
    top_product = master_df.groupby("product_category_name_english")["price"].sum().idxmax()
    total_orders = master_df["order_id"].nunique()
    avg_churn = rfm["churn_probability"].mean()

    metrics_summary = f"""
    You are a business intelligence analyst for a Nigerian e-commerce company similar to Jumia.
    Analyze the following business metrics and generate clear, actionable insights in plain English.

    REPORT PERIOD: {datetime.now().strftime('%B %Y')}

    REVENUE SUMMARY:
    - Total Revenue: {total_revenue:,.2f}
    - Total Orders: {total_orders}
    - Average Delivery Time: {avg_delivery:.1f} days
    - Top Performing State: {top_state}
    - Top Performing Product: {top_product}

    CUSTOMER SEGMENTS:
    {rfm['segment'].value_counts().to_string()}

    CHURN SUMMARY:
    - Average Churn Probability: {avg_churn:.2f}
    - Total Churned Customers: {rfm[rfm['segment'] == 'Churned'].shape[0]}

    Generate 5 key business insights and 3 actionable recommendations based on this data.
    Write in plain English that a business owner can understand without technical knowledge.
    """

    client = Groq(api_key="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")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": metrics_summary}]
    )

    insights = response.choices[0].message.content
    print("\n=== AI GENERATED INSIGHTS ===")
    print(insights)

    # Save insights to file
    insights_path = os.path.join(report_folder, "insights.txt")
    with open(insights_path, "w") as f:
        f.write(f"BUSINESS INTELLIGENCE REPORT — {datetime.now().strftime('%B %Y')}\n")
        f.write("="*50 + "\n\n")
        f.write(insights)
    print(f"\nInsights saved to {insights_path} ✅")

    # ============================================
    # STEP 9 — EXPORT FRESH CSV FOR DASHBOARD
    # ============================================
    print("\nStep 9 — Exporting fresh CSV...")
    master_df.to_csv("master_table.csv", index=False)
    rfm.to_csv("rfm_table.csv", index=False)
    print("CSV files exported ✅")

    # ============================================
    # STEP 10 — UPDATE SQL DATABASE
    # ============================================
    print("\nStep 10 — Updating SQL database...")
    try:
        password = quote_plus("Somide0987+")
        engine = create_engine(f"mysql+mysqlconnector://root:{password}@localhost/business", pool_pre_ping=True)
        engine.dispose()
        with engine.begin() as conn:
            master_df.to_sql("master_table", con=conn, if_exists="replace", index=False, chunksize=5000)
        print("SQL updated ✅")
    except Exception as e:
        print(f"SQL update failed: {e}")
    print(f"\n{'='*50}")
    print(f" PIPELINE COMPLETED — {datetime.now().strftime('%B %d, %Y %H:%M')}")
    print(f" Report saved to: {report_folder}")
    print(f"{'='*50}\n")

# ============================================
# FILE WATCHER — TRIGGERS ON NEW CSV
# ============================================

class NewDataHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".csv"):
            print(f"\n New file detected: {event.src_path}")
            print("Triggering pipeline automatically...")
            time.sleep(2)
            run_pipeline()


# ============================================
# START THE WATCHER
# ============================================

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    watch_folder = "."
    print(f"{'='*50}")
    print(f" BI PIPELINE WATCHER STARTED")
    print(f" Watching folder: {watch_folder}")
    print(f" Drop a new CSV file to trigger automatically")
    print(f"{'='*50}\n")

    event_handler = NewDataHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nPipeline watcher stopped.")
    observer.join()
