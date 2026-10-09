
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("models/churn_model.joblib")
DATA_PATH = Path("data/processed/customer_features.csv")
METRICS_PATH = Path("reports/final_evaluation_metrics.csv")
CONFUSION_PATH = Path("reports/confusion_matrix.csv")
COEFFICIENTS_PATH = Path("reports/model_coefficients.csv")
THRESHOLD_PATH = Path("reports/threshold_analysis.csv")

THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.70

PANEL = "#131E30"
TEAL = "#36D6C3"
BLUE = "#6EA8FE"
CORAL = "#FF7F87"
PURPLE = "#B9A0FF"
TEXT = "#E8EEF8"
MUTED = "#9BAAC0"

st.markdown("""
<style>
.stApp {
    background: #0B1220;
    color: #E8EEF8;
}
[data-testid="stSidebar"] {
    background: #101A2B;
    border-right: 1px solid #24334A;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1550px;
}
h1, h2, h3 {
    color: #F3F7FF !important;
    letter-spacing: -0.4px;
}
p, label, .stMarkdown {
    color: #D5DFEF;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17263B, #111B2B);
    border: 1px solid #293B54;
    padding: 19px 20px;
    border-radius: 15px;
    min-height: 125px;
}
[data-testid="stMetricLabel"] {
    color: #9BAAC0 !important;
}
[data-testid="stMetricValue"] {
    color: #F4F8FF !important;
}
div.stButton > button {
    background: #36D6C3;
    color: #081521;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 44px;
}
div.stButton > button:hover {
    background: #65E5D5;
    color: #081521;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #172338;
    border-color: #34445D;
}
div[data-testid="stTabs"] button {
    font-weight: 600;
}
hr {
    border-color: #29384D;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #29384D;
    border-radius: 10px;
    overflow: hidden;
}
.hero {
    background: linear-gradient(120deg, #152B42 0%, #123B48 100%);
    border: 1px solid #285466;
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 25px;
}
.hero-label {
    color: #65E5D5;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-title {
    color: #F5F8FF;
    font-size: 31px;
    font-weight: 750;
    margin-top: 8px;
    line-height: 1.25;
}
.hero-subtitle {
    color: #C0D0E3;
    font-size: 15px;
    margin-top: 10px;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_customer_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_optional_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def chart_layout(fig, title, height=430):
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 19, "color": TEXT},
            "x": 0.03
        },
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": TEXT, "family": "Arial"},
        margin={"l": 45, "r": 25, "t": 75, "b": 50},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": TEXT}
        },
        hoverlabel={
            "bgcolor": "#17263B",
            "bordercolor": TEAL,
            "font": {"color": TEXT, "size": 13}
        }
    )
    fig.update_xaxes(
        gridcolor="#29384D",
        zerolinecolor="#34445D",
        linecolor="#34445D",
        tickfont={"color": MUTED}
    )
    fig.update_yaxes(
        gridcolor="#29384D",
        zerolinecolor="#34445D",
        linecolor="#34445D",
        tickfont={"color": MUTED}
    )
    return fig


def show_chart(fig):
    st.plotly_chart(fig, use_container_width=True)


def format_risk(value):
    if value >= HIGH_RISK_THRESHOLD:
        return "High"
    if value >= THRESHOLD:
        return "Medium"
    return "Low"


if not MODEL_PATH.exists():
    st.error("The trained model was not found. Run python src/train_model.py first.")
    st.stop()

if not DATA_PATH.exists():
    st.error("The customer feature dataset was not found. Run python src/build_customer_features.py first.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    customers = load_customer_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Unable to load the model or dataset: {error}")
    st.stop()

required_columns = [
    "CustomerID",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Active_Days",
    "Recency",
    "Average_Order_Value",
    "Churn"
]

missing_columns = [
    column for column in required_columns
    if column not in customers.columns
]

if missing_columns:
    st.error(f"Required columns are missing: {', '.join(missing_columns)}")
    st.stop()

feature_columns = list(model.feature_names_in_)
missing_features = [
    column for column in feature_columns
    if column not in customers.columns
]

if missing_features:
    st.error(f"Model input features are missing: {', '.join(missing_features)}")
    st.stop()

customers = customers.copy()
customers["Churn"] = customers["Churn"].astype(int)

customers["Churn_Probability"] = model.predict_proba(
    customers[feature_columns]
)[:, 1]

customers["Predicted_Churn"] = (
    customers["Churn_Probability"] >= THRESHOLD
).astype(int)

customers["Risk_Band"] = customers["Churn_Probability"].apply(format_risk)

total_customers = len(customers)
churned_customers = int(customers["Churn"].sum())
retained_customers = total_customers - churned_customers
churn_rate = churned_customers / total_customers if total_customers else 0
total_revenue = customers["Monetary"].sum()
average_spending = customers["Monetary"].mean()
average_recency = customers["Recency"].mean()

high_risk_count = int((customers["Risk_Band"] == "High").sum())
medium_risk_count = int((customers["Risk_Band"] == "Medium").sum())
low_risk_count = int((customers["Risk_Band"] == "Low").sum())

st.markdown("""
<div class="hero">
    <div class="hero-label">CUSTOMER ANALYTICS PLATFORM</div>
    <div class="hero-title">Customer Retention Intelligence</div>
    <div class="hero-subtitle">
        Understand customer behavior, identify churn risk, and discover
        opportunities to improve retention.
    </div>
</div>
""", unsafe_allow_html=True)

tab_analytics, tab_risk, tab_predict, tab_model = st.tabs([
    "📈 Analytics Overview",
    "🎯 High-Risk Customers",
    "🔮 Predict Churn",
    "🧪 Model Performance"
])

with tab_analytics:
    st.subheader("Business Overview")
    st.caption("Customer metrics calculated from the prepared historical dataset.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Historical Churn Rate", f"{churn_rate:.1%}")
    col3.metric("Total Customer Revenue", f"{total_revenue:,.2f}")
    col4.metric("Average Customer Spend", f"{average_spending:,.2f}")

    st.divider()
    st.subheader("Customer Behavior")

    chart1, chart2 = st.columns(2)

    with chart1:
        churn_summary = pd.DataFrame({
            "Customer Status": ["Retained", "Churned"],
            "Customers": [retained_customers, churned_customers]
        })
        fig = px.bar(
            churn_summary,
            x="Customer Status",
            y="Customers",
            color="Customer Status",
            color_discrete_map={"Retained": TEAL, "Churned": CORAL},
            text="Customers"
        )
        fig.update_traces(
            texttemplate="%{y:,}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<extra></extra>"
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Number of Customers")
        show_chart(chart_layout(fig, "Retained vs. Churned Customers"))

    with chart2:
        churn_pie = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [retained_customers, churned_customers]
        })
        fig = px.pie(
            churn_pie,
            names="Status",
            values="Customers",
            hole=0.62,
            color="Status",
            color_discrete_map={"Retained": TEAL, "Churned": CORAL}
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>"
        )
        show_chart(chart_layout(fig, "Historical Customer Distribution"))

    st.divider()
    st.subheader("Spending and Purchase Insights")

    chart3, chart4 = st.columns(2)

    with chart3:
        spending_limit = customers["Monetary"].quantile(0.95)
        spending = customers.copy()
        spending["Displayed Spending"] = spending["Monetary"].clip(
            upper=spending_limit
        )
        fig = px.histogram(
            spending,
            x="Displayed Spending",
            nbins=30,
            color_discrete_sequence=[BLUE],
            labels={"Displayed Spending": "Customer Spending"},
            hover_data={"Monetary": ":,.2f"}
        )
        fig.update_traces(
            marker_line_color=PANEL,
            marker_line_width=1,
            hovertemplate="Spending range: %{x:,.2f}<br>Customers: %{y}<extra></extra>"
        )
        fig.update_xaxes(title="Total Spending (display capped at 95th percentile)")
        fig.update_yaxes(title="Number of Customers")
        show_chart(chart_layout(fig, "Customer Spending Distribution"))

    with chart4:
        scatter_data = customers.copy()
        scatter_data["Historical Status"] = scatter_data["Churn"].map({
            0: "Retained",
            1: "Churned"
        })
        fig = px.scatter(
            scatter_data,
            x="Frequency",
            y="Monetary",
            color="Historical Status",
            color_discrete_map={"Retained": TEAL, "Churned": CORAL},
            hover_data={
                "CustomerID": True,
                "Frequency": ":,",
                "Monetary": ":,.2f",
                "Recency": True,
                "Active_Days": True,
                "Historical Status": True
            },
            log_x=True,
            log_y=True,
            opacity=0.75,
            labels={
                "Frequency": "Number of Orders",
                "Monetary": "Total Spending",
                "Historical Status": "Customer Status"
            }
        )
        show_chart(chart_layout(fig, "Orders vs. Customer Spending"))

    st.divider()
    st.subheader("Purchase Activity")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Orders per Customer", f"{customers['Frequency'].mean():.1f}")
    col2.metric("Average Active Days", f"{customers['Active_Days'].mean():.1f}")
    col3.metric("Average Days Since Last Purchase", f"{average_recency:.1f}")

    st.caption(
        "Historical churn means no purchase during the defined label window. "
        "It does not prove that a customer will never return."
    )

with tab_risk:
    st.subheader("Customer Churn Risk Prioritization")
    st.caption(
        "Risk groups are based on model-estimated probabilities. "
        "They do not guarantee future customer behavior."
    )

    risk_col1, risk_col2, risk_col3 = st.columns(3)
    risk_col1.metric("High Risk · ≥70%", f"{high_risk_count:,}")
    risk_col2.metric("Medium Risk · 40–69.9%", f"{medium_risk_count:,}")
    risk_col3.metric("Low Risk · <40%", f"{low_risk_count:,}")

    st.divider()

    risk_filter = st.selectbox(
        "Filter by risk category",
        [
            "All customers",
            "High risk",
            "Medium risk",
            "Low risk",
            "Historically churned only"
        ]
    )

    minimum_spend = float(customers["Monetary"].min())
    maximum_spend = float(customers["Monetary"].max())

    spending_range = st.slider(
        "Filter by total spending",
        min_value=minimum_spend,
        max_value=maximum_spend,
        value=(minimum_spend, maximum_spend)
    )

    risk_customers = customers[
        customers["Monetary"].between(
            spending_range[0],
            spending_range[1]
        )
    ].copy()

    if risk_filter == "High risk":
        risk_customers = risk_customers[risk_customers["Risk_Band"] == "High"]
    elif risk_filter == "Medium risk":
        risk_customers = risk_customers[risk_customers["Risk_Band"] == "Medium"]
    elif risk_filter == "Low risk":
        risk_customers = risk_customers[risk_customers["Risk_Band"] == "Low"]
    elif risk_filter == "Historically churned only":
        risk_customers = risk_customers[risk_customers["Churn"] == 1]

    risk_customers = risk_customers.sort_values(
        "Churn_Probability",
        ascending=False
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Customers Above 40% Threshold",
        f"{int((customers['Predicted_Churn'] == 1).sum()):,}"
    )
    col2.metric("Customers in Current View", f"{len(risk_customers):,}")
    col3.metric(
        "Mean Risk Probability",
        f"{risk_customers['Churn_Probability'].mean():.1%}"
        if len(risk_customers) else "N/A"
    )

    st.subheader("Highest-Risk Customers")

    risk_chart_data = risk_customers.head(20).copy()
    if not risk_chart_data.empty:
        risk_chart_data["Customer Label"] = risk_chart_data["CustomerID"].astype(str)
        fig = px.bar(
            risk_chart_data.sort_values("Churn_Probability"),
            x="Churn_Probability",
            y="Customer Label",
            orientation="h",
            color="Churn_Probability",
            color_continuous_scale=["#36D6C3", "#6EA8FE", "#FF7F87"],
            hover_data={
                "CustomerID": True,
                "Monetary": ":,.2f",
                "Frequency": True,
                "Recency": True,
                "Churn_Probability": ":.1%"
            },
            labels={
                "Churn_Probability": "Predicted Churn Probability",
                "Customer Label": "Customer ID",
                "Monetary": "Total Spending",
                "Frequency": "Number of Orders",
                "Recency": "Days Since Purchase"
            }
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".0%")
        show_chart(chart_layout(fig, "Top 20 Customers by Predicted Risk", 550))

    table_columns = [
        "CustomerID",
        "Churn_Probability",
        "Risk_Band",
        "Monetary",
        "Frequency",
        "Recency",
        "Active_Days",
        "Predicted_Churn",
        "Churn"
    ]

    display_df = risk_customers[table_columns].head(100).copy()
    display_df = display_df.rename(columns={
        "CustomerID": "Customer ID",
        "Churn_Probability": "Churn Probability",
        "Risk_Band": "Risk Category",
        "Monetary": "Total Spending",
        "Frequency": "Number of Orders",
        "Recency": "Days Since Purchase",
        "Active_Days": "Active Days",
        "Predicted_Churn": "Above 40% Threshold",
        "Churn": "Historical Churn Label"
    })

    if not display_df.empty:
        display_df["Churn Probability"] = display_df["Churn Probability"].map(
            lambda value: f"{value:.1%}"
        )
        display_df["Total Spending"] = display_df["Total Spending"].map(
            lambda value: f"{value:,.2f}"
        )
        display_df["Above 40% Threshold"] = display_df[
            "Above 40% Threshold"
        ].map({1: "Yes", 0: "No"})
        display_df["Historical Churn Label"] = display_df[
            "Historical Churn Label"
        ].map({1: "Churned", 0: "Retained"})

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = risk_customers[table_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Customer Risk Report",
        data=csv_data,
        file_name="customer_churn_risk_report.csv",
        mime="text/csv"
    )

    st.caption(
        "The table displays up to 100 matching customers, sorted by predicted "
        "churn probability. Historical labels are included for retrospective analysis."
    )

with tab_predict:
    st.subheader("Individual Customer Prediction")
    st.caption("Enter customer behavior details to estimate churn probability.")

    with st.form("customer_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            frequency = st.number_input(
                "Number of Orders",
                min_value=1,
                max_value=10000,
                value=5
            )
            monetary = st.number_input(
                "Total Spending",
                min_value=0.0,
                value=1000.0,
                step=100.0
            )
            total_quantity = st.number_input(
                "Total Quantity Purchased",
                min_value=1,
                value=100
            )
            unique_products = st.number_input(
                "Unique Products Purchased",
                min_value=1,
                value=20
            )

        with col2:
            active_days = st.number_input(
                "Active Purchase Days",
                min_value=1,
                value=10
            )
            recency = st.number_input(
                "Days Since Last Purchase",
                min_value=0,
                value=30
            )
            average_order_value = monetary / frequency
            st.metric("Average Order Value", f"{average_order_value:,.2f}")

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:
        customer_input = pd.DataFrame([{
            "Frequency": frequency,
            "Monetary": monetary,
            "Total_Quantity": total_quantity,
            "Unique_Products": unique_products,
            "Active_Days": active_days,
            "Recency": recency,
            "Average_Order_Value": average_order_value
        }])[feature_columns]

        probability = float(model.predict_proba(customer_input)[0, 1])
        predicted_churn = probability >= THRESHOLD
        risk_band = format_risk(probability)

        st.divider()
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Churn Probability", f"{probability:.1%}")
        col2.metric("Decision Threshold", f"{THRESHOLD:.0%}")
        col3.metric("Risk Category", risk_band)

        st.progress(probability)

        if risk_band == "High":
            st.error("High Churn Risk")
            st.write(
                "Suggested action: Prioritize personalized outreach and review "
                "whether a relevant retention offer may help."
            )
        elif predicted_churn:
            st.warning("Medium Churn Risk")
            st.write(
                "Suggested action: Review recent activity and consider "
                "targeted engagement."
            )
        else:
            st.success("Low Churn Risk")
            st.write(
                "Suggested action: Maintain engagement and monitor future "
                "purchase activity."
            )

        st.divider()
        st.subheader("Factors Associated with This Prediction")

        explanation_rows = [
            {
                "Feature": "Active Days",
                "Customer Value": active_days,
                "Model Association": "More active purchase days are associated with lower predicted churn."
            },
            {
                "Feature": "Unique Products",
                "Customer Value": unique_products,
                "Model Association": "More unique products are associated with lower predicted churn."
            },
            {
                "Feature": "Total Spending",
                "Customer Value": monetary,
                "Model Association": "Higher historical spending is associated with lower predicted churn."
            },
            {
                "Feature": "Average Order Value",
                "Customer Value": average_order_value,
                "Model Association": "Higher average order value is associated with higher predicted churn in this model."
            },
            {
                "Feature": "Days Since Last Purchase",
                "Customer Value": recency,
                "Model Association": "More days since the last purchase are associated with higher predicted churn."
            },
            {
                "Feature": "Total Quantity",
                "Customer Value": total_quantity,
                "Model Association": "Higher total quantity is associated with higher predicted churn in this model."
            },
            {
                "Feature": "Number of Orders",
                "Customer Value": frequency,
                "Model Association": "Order frequency has a relatively small positive coefficient in this model."
            }
        ]

        st.dataframe(
            pd.DataFrame(explanation_rows),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are general model associations, not exact explanations "
            "of this individual prediction. They do not establish causation."
        )

        st.caption(
            "This prediction is an estimate based on historical patterns. "
            "It should support, not replace, business judgment."
        )

with tab_model:
    st.subheader("Model Performance")
    st.caption(
        "Evaluation metrics are read from your saved report. "
        "Check the report's test split and threshold before interpreting the results."
    )

    metrics = load_optional_csv(str(METRICS_PATH))
    confusion = load_optional_csv(str(CONFUSION_PATH))
    coefficients = load_optional_csv(str(COEFFICIENTS_PATH))
    threshold_data = load_optional_csv(str(THRESHOLD_PATH))

    if not metrics.empty:
        metric_row = metrics.iloc[0]

        metric_cols = st.columns(5)
        metric_names = [
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1 Score", "F1_Score"),
            ("ROC-AUC", "ROC_AUC")
        ]

        for column, (label, key) in zip(metric_cols, metric_names):
            if key in metrics.columns and pd.notna(metric_row[key]):
                column.metric(label, f"{float(metric_row[key]):.1%}")
            else:
                column.metric(label, "N/A")

        if "Decision_Threshold" in metrics.columns:
            st.caption(
                f"Saved decision threshold: "
                f"{float(metric_row['Decision_Threshold']):.0%}"
            )
    else:
        st.warning(
            "Evaluation metrics report not found. Run your model evaluation "
            "and save the results to reports/final_evaluation_metrics.csv."
        )

    st.divider()
    st.subheader("Confusion Matrix")

    if not confusion.empty:
        numeric_confusion = confusion.select_dtypes(include="number")

        if numeric_confusion.shape == (2, 2):
            matrix_values = numeric_confusion.to_numpy()
            fig = go.Figure(data=go.Heatmap(
                z=matrix_values,
                x=["Predicted Retained", "Predicted Churned"],
                y=["Actual Retained", "Actual Churned"],
                colorscale=[
                    [0, "#17263B"],
                    [0.5, "#2A7185"],
                    [1, TEAL]
                ],
                text=matrix_values,
                texttemplate="%{text:,}",
                textfont={"size": 20, "color": TEXT},
                hovertemplate=(
                    "%{y}<br>%{x}<br>Customers: %{z:,}<extra></extra>"
                ),
                showscale=False
            ))
            show_chart(chart_layout(fig, "Confusion Matrix", 370))
        else:
            st.dataframe(confusion, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No saved confusion matrix was found at reports/confusion_matrix.csv."
        )

    st.caption(
        "True negatives are correctly identified retained customers; "
        "true positives are correctly identified churned customers. "
        "False positives and false negatives represent classification errors."
    )

    st.divider()
    st.subheader("Model Coefficients")

    if not coefficients.empty:
        feature_col = next(
            (
                column for column in coefficients.columns
                if column.lower() in ["feature", "feature_name", "features"]
            ),
            None
        )
        coefficient_col = next(
            (
                column for column in coefficients.columns
                if "coefficient" in column.lower() or column.lower() == "coef"
            ),
            None
        )

        if feature_col and coefficient_col:
            coefficient_data = coefficients.copy()
            coefficient_data[coefficient_col] = pd.to_numeric(
                coefficient_data[coefficient_col],
                errors="coerce"
            )
            coefficient_data = coefficient_data.dropna(
                subset=[coefficient_col]
            ).sort_values(coefficient_col)

            coefficient_data["Direction"] = coefficient_data[
                coefficient_col
            ].apply(
                lambda value: "Positive association"
                if value > 0 else "Negative association"
            )

            fig = px.bar(
                coefficient_data,
                x=coefficient_col,
                y=feature_col,
                orientation="h",
                color="Direction",
                color_discrete_map={
                    "Positive association": CORAL,
                    "Negative association": TEAL
                },
                hover_data={
                    coefficient_col: ":.4f",
                    "Direction": True
                },
                labels={
                    coefficient_col: "Model Coefficient",
                    feature_col: "Feature"
                }
            )
            fig.add_vline(x=0, line_color=MUTED, line_dash="dash")
            show_chart(chart_layout(fig, "Logistic Regression Coefficients", 450))
            st.caption(
                "Coefficient direction indicates association with the model's "
                "churn score, holding other inputs fixed. It does not establish causation."
            )
        else:
            st.dataframe(coefficients, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No coefficient report found. Run python src/analyze_model.py first."
        )

    st.divider()
    st.subheader("Decision Threshold Analysis")

    if not threshold_data.empty and {
        "Threshold", "Precision", "Recall", "F1_Score"
    }.issubset(threshold_data.columns):
        plot_data = threshold_data.melt(
            id_vars=["Threshold"],
            value_vars=["Precision", "Recall", "F1_Score"],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.line(
            plot_data,
            x="Threshold",
            y="Score",
            color="Metric",
            markers=True,
            color_discrete_map={
                "Precision": CORAL,
                "Recall": TEAL,
                "F1_Score": BLUE
            },
            hover_data={"Threshold": ":.2f", "Score": ":.3f"},
            labels={"Score": "Metric Score"}
        )
        fig.update_xaxes(tickformat=".2f")
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        fig.add_vline(
            x=THRESHOLD,
            line_dash="dash",
            line_color=PURPLE,
            annotation_text="Current 40% threshold",
            annotation_font_color=PURPLE
        )
        show_chart(chart_layout(fig, "Precision, Recall and F1 by Threshold", 430))
        st.caption(
            "Changing the threshold changes the balance between missed churners "
            "and customers flagged for intervention. Threshold tuning should be "
            "performed on validation data, not the final test set."
        )
    else:
        st.info(
            "No usable threshold report found at reports/threshold_analysis.csv."
        )

st.divider()

st.markdown(
    "<div style='text-align:center;color:#71839D;font-size:12px;'>"
    "CUSTOMER RETENTION INTELLIGENCE · MACHINE LEARNING ANALYTICS"
    "</div>",
    unsafe_allow_html=True
)
