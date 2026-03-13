import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
import plotly.express as px

st.set_page_config(page_title="DecideAI",layout="wide")

st.title("DecideAI – Autonomous Agentic Decision Intelligence System")

st.caption("Transforming Raw Data into Autonomous AI Decisions")

st.write("Upload any dataset and DecideAI will automatically analyze, predict and generate decision insights")

uploaded_file=st.file_uploader("Upload CSV file",type=["csv"])

if uploaded_file is not None:

    df=pd.read_csv(uploaded_file)

    st.subheader("Data Intelligence Module")

    st.dataframe(df.head())

    # DATA CLEANING

    for col in df.columns:

        if df[col].dtype=="object":

            df[col]=df[col].fillna(df[col].mode()[0])

        else:

            df[col]=df[col].fillna(df[col].mean())

    c1,c2,c3=st.columns(3)

    with c1:
        st.write("Rows:",df.shape[0])

    with c2:
        st.write("Columns:",df.shape[1])

    with c3:
        st.write("Missing Fixed:",df.isnull().sum().sum())

    # DATASET QUALITY

    st.subheader("Dataset Quality Engine")

    missing=df.isnull().sum().sum()

    quality=100-(missing*100/(df.shape[0]*df.shape[1]))

    st.write("Dataset Quality Score:",round(quality,2),"%")

    target=st.selectbox("Select Target Column",df.columns)

    X=df.drop(target,axis=1)

    y=df[target]

    # ENCODING

    le=LabelEncoder()

    for col in X.columns:

        if X[col].dtype=="object":

            X[col]=le.fit_transform(X[col].astype(str))

    if y.dtype=="object":

        y=le.fit_transform(y.astype(str))

    # PROBLEM DETECTION

    unique=len(set(y))

    if unique<20:

        problem="Classification"

    else:

        problem="Regression"

    st.subheader("AI Problem Detection Engine")

    st.success(problem)

    X_train,X_test,y_train,y_test=train_test_split(

        X,y,test_size=0.2,random_state=42

    )

    st.subheader("AI Processing Pipeline")

    st.info("Step 1 Data Cleaning Completed")

    st.info("Step 2 Problem Detection Completed")

    st.info("Step 3 Model Training Running")

    # AUTOML

    st.subheader("AutoML Engine")

    if problem=="Classification":

        rf=RandomForestClassifier()

        rf.fit(X_train,y_train)

        rf_score=rf.score(X_test,y_test)

        lr=LogisticRegression(max_iter=1000)

        lr.fit(X_train,y_train)

        lr_score=lr.score(X_test,y_test)

    else:

        rf=RandomForestRegressor()

        rf.fit(X_train,y_train)

        rf_score=rf.score(X_test,y_test)

        lr=LinearRegression()

        lr.fit(X_train,y_train)

        lr_score=lr.score(X_test,y_test)

    st.write("RandomForest Score:",rf_score)

    st.write("Linear/Logistic Score:",lr_score)

    # MODEL COMPARISON

    model_df=pd.DataFrame({

        "Model":["RandomForest","Linear/Logistic"],

        "Score":[rf_score,lr_score]

    })

    fig=px.bar(

        model_df,

        x="Model",

        y="Score",

        title="Model Performance Comparison"

    )

    st.plotly_chart(fig)

    if rf_score>lr_score:

        best_model=rf

        best_score=rf_score

        model_name="Random Forest"

    else:

        best_model=lr

        best_score=lr_score

        model_name="Linear/Logistic"

    st.success("Selected Model:"+model_name)

    st.success("Performance Score:"+str(round(best_score,3)))

    # CONFIDENCE

    confidence=round(best_score*100,2)

    st.subheader("AI Confidence Engine")

    st.write("Confidence:",confidence,"%")

    # RISK ENGINE

    st.subheader("AI Risk Engine")

    if best_score>0.85:

        risk="LOW"

    elif best_score>0.65:

        risk="MEDIUM"

    else:

        risk="HIGH"

    st.write("Risk Level:",risk)

    if risk=="HIGH":

        st.error("High uncertainty detected")

    elif risk=="MEDIUM":

        st.warning("Moderate reliability")

    else:

        st.success("Stable prediction patterns")

    # AI RECOMMENDATION SCORE

    st.subheader("AI Recommendation Strength")

    if confidence>85:

        st.success("AI strongly recommends using predictions")

    elif confidence>70:

        st.warning("AI moderately recommends usage")

    else:

        st.error("AI suggests improving dataset")

    # EXPLAINABLE AI

    if model_name=="Random Forest":

        importance=best_model.feature_importances_

        imp=pd.DataFrame({

            "Feature":X.columns,

            "Importance":importance

        })

        imp=imp.sort_values(

            by="Importance",

            ascending=False

        )

        st.subheader("Explainable AI Decision Factors")

        fig=px.bar(

            imp.head(10),

            x="Feature",

            y="Importance",

            title="Top Decision Drivers"

        )

        st.plotly_chart(fig)

        st.subheader("AI Strategy Engine")

        top=imp.head(3)

        for i,row in top.iterrows():

            st.write(

            "Improve",row["Feature"],

            "to improve outcomes"

            )

    # DECISION SIMULATOR

    st.subheader("Decision Simulator")

    user_input={}

    for col in X.columns:

        user_input[col]=st.number_input(

            "Adjust "+col,

            value=float(X[col].mean())

        )

    if st.button("Run AI Decision"):

        input_df=pd.DataFrame([user_input])

        pred=best_model.predict(input_df)

        st.success("Predicted Result:"+str(pred[0]))

    # TOP INSIGHTS

    st.subheader("Top AI Insights")

    st.write("• Key variables strongly impact prediction")

    st.write("• Model stability depends on data consistency")

    st.write("• Improving top factors improves outcomes")

    st.write("• Simulator helps test strategies")

    # DATASET INSIGHT

    st.subheader("AI Dataset Intelligence")

    insight=f"""

Dataset size {df.shape[0]} rows.

{df.shape[1]} variables analyzed.

Problem type {problem}.

Best model {model_name}.

Confidence {confidence}%.

Risk level {risk}.

AI recommends focusing on key decision variables.

"""

    st.write(insight)

    # AGENT REASONING

    st.subheader("Agentic AI Reasoning Engine")

    reasoning="""

AI Agent Workflow:

Data cleaned automatically.

Problem detected automatically.

Models trained automatically.

Best model selected.

Decision factors identified.

Risk assessed.

Strategies generated.

AI completed autonomous decision analysis.

"""

    st.write(reasoning)

    # USE CASES

    st.subheader("Real World Applications")

    st.write("Education → Student prediction")

    st.write("Finance → Risk prediction")

    st.write("Healthcare → Outcome prediction")

    st.write("HR → Performance prediction")

    st.write("Business → Decision optimization")

    # FUTURE SCOPE

    st.subheader("Future Enhancements")

    st.write("LLM integration")

    st.write("Real time data")

    st.write("Deep learning")

    st.write("Reinforcement learning")

    st.write("Multi agent AI")

    # SYSTEM STATUS

    st.subheader("AI System Status")

    st.success("Data Engine Active")

    st.success("AutoML Engine Active")

    st.success("Explainable AI Active")

    st.success("Decision Engine Active")

    st.success("Agent Reasoning Active")

    st.success("System Operational")

    # FINAL REPORT

    st.subheader("Autonomous AI Agent Report")

    report=f"""

DecideAI Final Report

Rows:
{df.shape[0]}

Columns:
{df.shape[1]}

Problem:
{problem}

Best Model:
{model_name}

Score:
{round(best_score,3)}

Confidence:
{confidence}%

Risk:
{risk}

Status:
AI Analysis Completed

"""

    st.write(report)