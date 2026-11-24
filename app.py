import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier

from simulation import (
    generate_simulated_servers,
    evolve_simulated_servers,
)


# LOAD SECURITY CONFIGURATION
load_dotenv()
ENROLLMENT_KEY = os.getenv("CLOUDWISE_ENROLLMENT_KEY")


# ==================================================
# CONNECTED DEVICE FUNCTION
# ==================================================

def get_connected_devices():

    if not ENROLLMENT_KEY:
        return []

    try:
        response = requests.get(
            "http://127.0.0.1:8000/devices",
            headers={
                "x-enrollment-key": ENROLLMENT_KEY
            },
            timeout=3
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as error:
        print("Device connection error:", error)
        return []

# ==================================================
# GET DEVICE HISTORICAL METRICS
# ==================================================

def get_device_history(device_id, limit=100):

    if not ENROLLMENT_KEY:
        return []

    try:
        response = requests.get(
            f"http://127.0.0.1:8000/history/{device_id}",
            params={"limit": limit},
            headers={
                "x-enrollment-key": ENROLLMENT_KEY
            },
            timeout=5
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as error:
        print("History connection error:", error)
        return []
    
# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CloudWise AI",
    page_icon="☁️",
    layout="wide",
)


# ==================================================
# LOAD MODEL AND DATASET
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("cloudwise_model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_csv("data/Cost_Optimization.csv")


model = load_model()
df = load_dataset()

FEATURES = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes",
]

TARGET = "recommendation"

CLASS_ORDER = [
    "Keep Running",
    "Shutdown",
    "Sleep Mode",
]


# ==================================================
# EVALUATE MODEL
# ==================================================

@st.cache_data
def evaluate_model():
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    evaluation_model = DecisionTreeClassifier(
        max_depth=6,
        random_state=42,
    )

    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_ORDER,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    return accuracy, matrix, report, len(y_test)
accuracy, matrix, report, test_samples = evaluate_model()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("☁️ CloudWise AI")

    st.caption("Cloud Resource Optimization Advisor")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Choose a section",
        [
            "Resource Advisor",
            "Model Performance",
            "Dataset Insights",
            "Cloud Cost Estimator",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Recommendation Types")

    st.write("🟢 **Keep Running**")
    st.write("🟡 **Sleep Mode**")
    st.write("🔴 **Shutdown**")

    st.divider()

    st.caption(
        "Student project | Experimental ML predictions"
    )


    st.divider()

@st.fragment(run_every="5s")
def connected_devices_panel():

        st.markdown("### 💻 Connected Devices")

        devices = get_connected_devices()

        if devices:

            for device in devices:

                if device["last_seen"]:

                    last_seen = datetime.fromisoformat(
                        device["last_seen"].replace("Z", "+00:00")
                    )

                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)

                time_difference = (
                    datetime.now(timezone.utc) - last_seen
                ).total_seconds()

                if time_difference <= 30:
                    st.success(f"🟢 Online: {device['device_name']}")
                else:
                    st.error(f"🔴 Offline: {device['device_name']}")

            else:
                st.warning("No activity recorded yet.")

                st.write(f"CPU Usage: {device['cpu_usage']}%")
                st.write(f"RAM Usage: {device['ram_usage']}%")
                st.write(f"Disk Usage: {device['disk_usage']}%")
                st.write(f"Active Processes: {device['active_jobs']}")

                # HISTORICAL MONITORING
                st.markdown("#### 📊 Resource Usage History")

                history = get_device_history(device["device_id"], limit=100)

                if history:

                    history_df = pd.DataFrame(history)

                    history_df["recorded_at"] = pd.to_datetime(
                        history_df["recorded_at"],
                        utc=True
                    )

                    history_df = history_df.sort_values("recorded_at")

                    history_df = history_df.set_index("recorded_at")

                    st.line_chart(
                        history_df[
                            ["cpu_usage", "ram_usage", "disk_usage"]
                        ]
                    )

                    # ==================================================
                    # RESOURCE ANALYTICS
                    # ==================================================

                    st.markdown("#### 📈 Resource Analytics")

                    avg_cpu = history_df["cpu_usage"].mean()
                    peak_cpu = history_df["cpu_usage"].max()
                    avg_ram = history_df["ram_usage"].mean()
                    avg_disk = history_df["disk_usage"].mean()

                    col1, col2, col3, col4 = st.columns(4)

                    col1, col2 = st.columns(2)

                    col1.metric("Average CPU", f"{avg_cpu:.2f} %")
                    col2.metric("Peak CPU", f"{peak_cpu:.2f} %")

                    col3, col4 = st.columns(2)

                    col3.metric("Average RAM", f"{avg_ram:.2f} %")
                    col4.metric("Average Disk", f"{avg_disk:.2f} %")
                else:
                    st.info("No historical records available yet.")

        else:
            st.warning("No devices connected.")

connected_devices_panel()
# ==================================================
# RECOMMENDATION DISPLAY FUNCTION
# ==================================================

def show_recommendation(prediction):

    st.header("🔍 AI Analysis Result")

    if prediction == "Keep Running":

        st.success(
            "🟢 Predicted Recommendation: Keep Running"
        )

        st.write(
            "The model classifies this resource scenario "
            "as Keep Running. Review workload demand and "
            "resource requirements before taking action."
        )

    elif prediction == "Sleep Mode":

        st.warning(
            "🟡 Predicted Recommendation: Sleep Mode"
        )

        st.write(
            "The model classifies this resource scenario "
            "as Sleep Mode. Check service availability, "
            "active workloads, and operational requirements."
        )

    elif prediction == "Shutdown":

        st.error(
            "🔴 Predicted Recommendation: Shutdown"
        )

        st.write(
            "The model classifies this resource scenario "
            "as Shutdown. Verify that no critical workloads "
            "or services depend on the resource."
        )

    st.info(
        "This is an experimental ML prediction based on "
        "the project dataset. No cloud server is modified "
        "or shut down automatically."
    )


# ==================================================
# PAGE 1: RESOURCE ADVISOR
# ==================================================

if page == "Resource Advisor":

    st.title("☁️ CloudWise AI")

    st.subheader(
        "AI-Powered Cloud Cost & Resource Optimization Advisor"
    )

    st.write(
        "Monitor cloud resources using manual inputs "
        "or automatic simulation."
    )

    st.divider()

    st.header("⚙️ Select Monitoring Mode")

    mode = st.radio(
        "Choose your monitoring mode:",
        [
            "Manual Mode",
            "Simulation Mode",
        ],
        horizontal=True,
    )

    st.divider()

    # ==================================================
    # MANUAL MODE
    # ==================================================

    if mode == "Manual Mode":

        st.header("🖥️ Manual Resource Monitoring")

        col1, col2 = st.columns(2)

        with col1:

            cpu = st.slider(
                "CPU Usage (%)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=0.5,
            )

            ram = st.slider(
                "RAM Usage (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=0.5,
            )

        with col2:

            jobs = st.number_input(
                "Number of Active Jobs",
                min_value=0,
                max_value=10000,
                value=2,
                step=1,
            )

            idle = st.number_input(
                "Idle Time (Minutes)",
                min_value=0.0,
                max_value=100000.0,
                value=20.0,
                step=1.0,
            )

        st.divider()

        st.header("📊 Current Input Metrics")

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric("CPU Usage", f"{cpu:.1f}%")
        metric2.metric("RAM Usage", f"{ram:.1f}%")
        metric3.metric("Active Jobs", f"{jobs:,}")
        metric4.metric("Idle Time", f"{idle:,.0f} min")

        st.divider()

        if st.button(
            "🤖 Generate AI Recommendation",
            type="primary",
            use_container_width=True,
        ):

            input_data = pd.DataFrame(
                [[cpu, ram, jobs, idle]],
                columns=FEATURES,
            )

            prediction = model.predict(input_data)[0]

            show_recommendation(prediction)

    # ==================================================
    # SIMULATION MODE
    # ==================================================

    elif mode == "Simulation Mode":

        st.header("🔄 Multi-Server Cloud Simulation")

        st.write(
            "Monitor 10 virtual servers with automatically "
            "changing resource metrics."
        )

        st.warning(
            "DEMO MODE: These are simulated server metrics. "
            "They are not measurements from real AWS servers."
        )

        # Initialize session state

        if "simulation_running" not in st.session_state:
            st.session_state["simulation_running"] = False

        if "simulated_servers" not in st.session_state:
            st.session_state["simulated_servers"] = []

        # Start simulation

        if not st.session_state["simulation_running"]:

            if st.button(
                "▶️ Start Simulation",
                type="primary",
                use_container_width=True,
            ):

                st.session_state["simulated_servers"] = (
                    generate_simulated_servers(df, count=10)
                )

                st.session_state["simulation_running"] = True

                st.rerun()

        else:

            st.success(
                "Simulation is running. Refresh interval: 5 seconds."
            )

        # Automatic dashboard refresh

        @st.fragment(
            run_every=(
                "5s"
                if st.session_state["simulation_running"]
                else None
            )
        )
        def simulation_dashboard():

            if not st.session_state["simulation_running"]:
                return

            servers = st.session_state["simulated_servers"]

            # Evolve virtual server metrics

            evolve_simulated_servers(servers, df)

            results = []

            # Generate ML predictions

            for server in servers:

                metrics = server["metrics"]

                input_data = pd.DataFrame(
                    [[
                        metrics["cpu_usage_percent"],
                        metrics["ram_usage_percent"],
                        metrics["active_jobs"],
                        metrics["idle_time_minutes"],
                    ]],
                    columns=FEATURES,
                )

                prediction = model.predict(input_data)[0]

                results.append({
                    "Server ID": server["server_id"],
                    "CPU (%)": metrics["cpu_usage_percent"],
                    "RAM (%)": metrics["ram_usage_percent"],
                    "Active Jobs": metrics["active_jobs"],
                    "Idle Time (min)": round(
                        metrics["idle_time_minutes"], 1
                    ),
                    "AI Recommendation": prediction,
                })

            results_df = pd.DataFrame(results)

            st.session_state["simulated_servers"] = servers

            st.divider()

            # Server summary

            st.subheader("📊 Server Overview")

            total_servers = len(results_df)

            keep_count = (
                results_df["AI Recommendation"] == "Keep Running"
            ).sum()

            sleep_count = (
                results_df["AI Recommendation"] == "Sleep Mode"
            ).sum()

            shutdown_count = (
                results_df["AI Recommendation"] == "Shutdown"
            ).sum()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Servers",
                total_servers,
            )

            col2.metric(
                "🟢 Keep Running",
                int(keep_count),
            )

            col3.metric(
                "🟡 Sleep Mode",
                int(sleep_count),
            )

            col4.metric(
                "🔴 Shutdown",
                int(shutdown_count),
            )

            st.divider()

            # Individual server monitoring

            st.subheader("🖥️ Individual Server Monitoring")

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # CPU and RAM charts

            st.subheader("📈 Server Resource Usage")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                st.write("CPU Usage (%)")

                st.bar_chart(
                    results_df.set_index("Server ID")[["CPU (%)"]]
                )

            with chart_col2:

                st.write("RAM Usage (%)")

                st.bar_chart(
                    results_df.set_index("Server ID")[["RAM (%)"]]
                )

            st.divider()

            st.caption(
                "Virtual metrics update every 5 seconds. "
                "AI recommendations are experimental predictions."
            )

            # Stop simulation

            if st.button(
                "⏹️ Stop Simulation",
                use_container_width=True,
            ):

                st.session_state["simulation_running"] = False

                st.rerun(scope="app")

        simulation_dashboard()


# ==================================================
# PAGE 2: MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.title("📈 Model Performance Dashboard")

    st.write(
    "Evaluation of a Decision Tree model trained on "
    "80% of the dataset and tested on the remaining "
    "20% using unseen test samples."
)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%",
    )

    col2.metric(
        "Test Samples",
        f"{test_samples:,}",
    )

    col3.metric(
        "Incorrect Predictions",
        int(matrix.sum() - matrix.trace()),
    )

    st.caption(
        "Accuracy is measured on the held-out dataset split. "
        "It does not establish performance on independent, "
        "real-world cloud workloads."
    )

    st.divider()

    st.header("🎯 Confusion Matrix")

    st.write(
        "Rows represent actual classes; columns represent "
        "the model's predicted classes. Diagonal values "
        "are correct predictions."
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(matrix, interpolation="nearest")

    ax.set_title("Model Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))

    ax.set_xticklabels(
        CLASS_ORDER,
        rotation=20,
        ha="right",
    )

    ax.set_yticklabels(CLASS_ORDER)

    fig.colorbar(
        image,
        ax=ax,
        label="Number of samples",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):

        for col in range(matrix.shape[1]):

            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, col] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    st.header("📋 Classification Report")

    report_data = []

    for class_name in CLASS_ORDER:

        class_metrics = report[class_name]

        report_data.append({
            "Class": class_name,
            "Precision": round(
                class_metrics["precision"], 4
            ),
            "Recall": round(
                class_metrics["recall"], 4
            ),
            "F1-Score": round(
                class_metrics["f1-score"], 4
            ),
            "Support": int(
                class_metrics["support"]
            ),
        })

    st.dataframe(
        pd.DataFrame(report_data),
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        "**Precision:** How many predictions for a class "
        "were correct."
    )

    st.write(
        "**Recall:** How many actual samples in a class "
        "the model identified correctly."
    )

    st.write(
        "**F1-score:** A combined measure of precision and recall."
    )

    st.divider()

    st.warning(
        "The dataset labels and their generation method need "
        "to be understood before interpreting high accuracy "
        "as evidence of real-world cloud optimization ability."
    )


# ==================================================
# PAGE 3: DATASET INSIGHTS
# ==================================================

elif page == "Dataset Insights":

    st.title("📊 Dataset Insights")

    st.write(
        "Explore the recommendation labels and resource "
        "metrics in the dataset used by this project."
    )

    st.divider()

    total_rows = len(df)

    total_columns = len(df.columns)

    missing_values = int(
        df.isnull().sum().sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Rows", f"{total_rows:,}")
    col2.metric("Dataset Columns", total_columns)
    col3.metric("Missing Values", missing_values)

    st.divider()

    st.header("Recommendation Distribution")

    counts = df[TARGET].value_counts()

    chart_col, table_col = st.columns(2)

    with chart_col:

        fig, ax = plt.subplots(figsize=(7, 5))

        counts.plot(
            kind="bar",
            ax=ax,
        )

        ax.set_title(
            "Number of Records by Recommendation"
        )

        ax.set_xlabel("Recommendation")
        ax.set_ylabel("Number of Records")

        ax.tick_params(
            axis="x",
            rotation=20,
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with table_col:

        distribution = pd.DataFrame({
            "Recommendation": counts.index,
            "Samples": counts.values,
            "Percentage": (
                counts.values / total_rows * 100
            ).round(2),
        })

        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.header("Resource Metric Summary")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    st.dataframe(
        df[numeric_columns].describe().round(2),
        use_container_width=True,
    )

    st.divider()

    st.header("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Dataset patterns describe the data available to this "
        "project. They are not live cloud measurements or "
        "verified cost-saving estimates."
    )


# ==================================================
# PAGE 4: CLOUD COST ESTIMATOR
# ==================================================

elif page == "Cloud Cost Estimator":

    st.title("💰 Cloud Cost Estimator")

    st.write(
        "Estimate possible cloud cost savings by entering "
        "your current monthly cost and an assumed savings "
        "percentage."
    )

    st.warning(
        "This is an illustrative calculator. It does not "
        "connect to AWS or any live cloud account, and its "
        "savings percentage is an assumption you choose."
    )

    st.divider()

    st.header("Enter Your Cloud Cost")

    current_monthly_cost = st.number_input(
        "Current Monthly Cloud Cost ($)",
        min_value=0.0,
        value=200.0,
        step=10.0,
    )

    savings_percentage = st.slider(
        "Assumed Savings Percentage (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
    )

    estimated_monthly_savings = (
        current_monthly_cost * savings_percentage / 100
    )

    estimated_new_cost = (
        current_monthly_cost - estimated_monthly_savings
    )

    estimated_annual_savings = (
        estimated_monthly_savings * 12
    )

    st.divider()

    st.header("📊 Estimated Cost Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Estimated Monthly Savings",
        f"${estimated_monthly_savings:,.2f}",
    )

    col2.metric(
        "Estimated Annual Savings",
        f"${estimated_annual_savings:,.2f}",
    )

    col3.metric(
        "Estimated New Monthly Cost",
        f"${estimated_new_cost:,.2f}",
    )

    st.divider()

    st.subheader("Cost Comparison")

    comparison = pd.DataFrame({
        "Cost Type": [
            "Current Monthly Cost",
            "Estimated Monthly Cost",
        ],
        "Amount ($)": [
            current_monthly_cost,
            estimated_new_cost,
        ],
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    comparison.plot(
        x="Cost Type",
        y="Amount ($)",
        kind="bar",
        ax=ax,
        legend=False,
    )

    ax.set_ylabel("Cost in USD ($)")
    ax.set_xlabel("")

    ax.set_title(
        "Current Cost vs Estimated Cost"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "The estimated savings are calculated only from "
        "the percentage you selected. They are not a prediction "
        "from the machine-learning model or verified savings "
        "from a cloud provider."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "CloudWise AI | Machine Learning Student Project | "
    "Experimental Resource Optimization Advisor"
)