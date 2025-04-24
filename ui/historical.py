import streamlit as st
import pandas as pd
import datetime

def create_historical_tab(vital_signs_collection):
    """Create the historical data tab"""
    st.subheader("Historical Data")
    
    # Add filters for MongoDB data
    st.subheader("Data Filters")
    col_date1, col_date2 = st.columns(2)
    
    with col_date1:
        start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
    
    with col_date2:
        end_date = st.date_input("End Date", datetime.date.today())
    
    # Convert to datetime
    start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
    end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
    
    # Query button
    if st.button("Query Database"):
        # Query MongoDB
        query_and_display_historical_data(vital_signs_collection, start_datetime, end_datetime)
    
    # Display current session data
    st.subheader("Current Session Data")
    display_session_data()

def query_and_display_historical_data(vital_signs_collection, start_datetime, end_datetime):
    """Query MongoDB for historical data and display it"""
    try:
        # Find data between dates
        query = {
            "timestamp": {
                "$gte": start_datetime,"$lte": end_datetime
            }
        }
        
        # Query the data
        results = list(vital_signs_collection.find(query).sort("timestamp", 1))
        
        # Convert to DataFrame
        if results:
            df_mongo = pd.DataFrame(results)
            # Drop MongoDB _id column
            if '_id' in df_mongo.columns:
                df_mongo = df_mongo.drop('_id', axis=1)
            
            # Display results
            st.dataframe(df_mongo)
            
            # Download option
            csv = df_mongo.to_csv(index=False)
            st.download_button(
                label="Download Query Results",
                data=csv,
                file_name=f"medical_data_{start_datetime.date()}_to_{end_datetime.date()}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data found for the selected date range.")
    except Exception as e:
        st.error(f"Error querying database: {e}")

def display_session_data():
    """Display current session data"""
    if 'df' in st.session_state and st.session_state.df is not None and len(st.session_state.df) > 0:
        st.dataframe(st.session_state.df)
        
        if st.button("Download Session Data"):
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="medical_session_data.csv",
                mime="text/csv"
            )
    else:
        st.info("No data recorded in this session yet. Start monitoring to collect data.")