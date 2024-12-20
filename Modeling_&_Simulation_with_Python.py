import streamlit as st



# Page Configuration
st.set_page_config(
    page_title="Modeling and Simulation with Python",
    page_icon="📘",
    layout="wide"
)


# Title
st.title("Modeling and Simulation with Python")
st.markdown("---")

# Introduction Section
st.markdown("<h3 style='text-align: center'>Introduction</h3>", unsafe_allow_html=True)
st.markdown(
    """
    Modeling and simulation have been integral to various aspects of human life throughout history, 
    offering powerful tools to understand, predict, and solve complex problems. As technology continues to evolve, 
    advancements in this field remain crucial for tackling future challenges. Through this project, we engaged in 
    practical applications of essential Python libraries and explored real-world modeling challenges by implementing 
    diverse machine learning algorithms and simulation techniques.
    """
)

st.markdown("<h3 style='text-align: center'>Project Overview</h3>", unsafe_allow_html=True)
st.markdown(
    """
    The project encompassed several key stages that showcase key aspects of Modeling & Simulation. Each important phase contributes to building a comprehensive understanding of machine learning processes, from data generation through model analysis and implementation.

    **Phase 1: Synthetic Data Generation and Preparation**📊  
    This phase focused on generating high-quality, customizable datasets tailored to specific modeling requirements. It involved creating multi-feature datasets, managing class distributions and feature parameters. This stage provided controlled environments essential for algorithm testing and ensured reliable training data.

    **Phase 2: Model Generation and Analysis**🤖  
    During this phase, we conducted exploratory data analysis (EDA) to understand data patterns and relationships. Various machine learning algorithms, including classical, ensemble, and neural network methods, were developed, tuned, and evaluated. The goal was to select and validate robust models using comprehensive performance metrics.

    **Phase 3: Model Implementation and Simulation**💻  
    The final phase centered on deploying the models into a practical simulation environment. This involved creating scalable prediction pipelines, real-time visualization, and scenario analysis tools. The results were analyzed to refine model performance and ensure applicability in real-world scenarios.
    """
)


# Footer or Additional Information
st.markdown("---")
with st.expander("By Team ORION"):
    st.write("""
        - Venn Delos P. Santos
        - John Christian M. Gava
        - Marc Christian D. Tumaneng
    """)

