import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px

def one_hot_encode_columns(X, categorical_columns):
    df_encoded = pd.get_dummies(X, columns=categorical_columns)
    return df_encoded

def quadratic_estimator(x, a, b, c):
    return a * x[0] + b * x[1] + c * x[0] * x[1]

def estimate_processing_time_quadratic(num_airports, date_range_days):
    a_optimized = -0.7982424463448559
    b_optimized = -1.7331784299486057
    c_optimized = 4.841257524103219

    return quadratic_estimator((num_airports, date_range_days), a_optimized, b_optimized, c_optimized)


def load_and_preprocess_data(selected_airports, start_date, end_date):
    # API key
    api_key = 'f0ccb5-b175f6'

    # URL template for the API endpoint
    api_endpoint_template = 'https://aviation-edge.com/v2/public/flightsHistory?key={}&code={}&type=departure&date_from={}&date_to={}'

    # Data list to store information across selected airports and dates
    data_list = []

    # Loop through each selected airport code
    for airport_code in selected_airports:
        # Loop through each date within the specified range
        current_date = pd.to_datetime(start_date)
        while current_date <= pd.to_datetime(end_date):
            # Convert the date to the required format
            formatted_date = current_date.strftime('%Y-%m-%d')

            # Construct the API endpoint URL
            api_endpoint = api_endpoint_template.format(api_key, airport_code, formatted_date, formatted_date)

            # Make a GET request to the API endpoint
            response = requests.get(api_endpoint)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the JSON data in the response
                flight_data = response.json()

                # Extract relevant information and append to the data list
                for flight in flight_data:
                    data_list.append({
                        'Type': flight.get('type'),
                        'Status': flight.get('status'),
                        'DepartureAirport': flight['departure'].get('iataCode'),
                        'DepartureIcaoCode': flight['departure'].get('icaoCode'),
                        'DepartureTerminal': flight['departure'].get('terminal'),
                        'DepartureGate': flight['departure'].get('gate'),
                        'DepartureDelay': flight['departure'].get('delay'),
                        'DepartureScheduledTime': flight['departure'].get('scheduledTime'),
                        'DepartureEstimatedTime': flight['departure'].get('estimatedTime'),
                        'DepartureActualTime': flight['departure'].get('actualTime'),
                        'DepartureEstimatedRunway': flight['departure'].get('estimatedRunway'),
                        'DepartureActualRunway': flight['departure'].get('actualRunway'),
                        'ArrivalAirport': flight['arrival'].get('iataCode'),
                        'ArrivalIcaoCode': flight['arrival'].get('icaoCode'),
                        'ArrivalBaggage': flight['arrival'].get('baggage'),
                        'ArrivalGate': flight['arrival'].get('gate'),
                        'ArrivalScheduledTime': flight['arrival'].get('scheduledTime'),
                        'ArrivalEstimatedTime': flight['arrival'].get('estimatedTime'),
                        'AirlineName': flight['airline'].get('name'),
                        'AirlineIataCode': flight['airline'].get('iataCode'),
                        'AirlineIcaoCode': flight['airline'].get('icaoCode'),
                        'FlightNumber': flight['flight'].get('number'),
                        'FlightIataNumber': flight['flight'].get('iataNumber'),
                        'FlightIcaoNumber': flight['flight'].get('icaoNumber')
                    })

            # Increment the date for the next iteration
            current_date += pd.Timedelta(days=1)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data_list)

    if not data_list:
        st.warning("Warning: No data available for the specified date range.")
        return None, None

    X = df[['Status', 'DepartureAirport', 'DepartureTerminal', 'DepartureGate', 'DepartureScheduledTime', 'DepartureEstimatedTime', 'DepartureEstimatedRunway', 'DepartureActualRunway', 'ArrivalAirport', 'AirlineName', 'FlightNumber', 'DepartureDelay']]

    X['DepartureEstimatedTime'] = pd.to_datetime(X['DepartureEstimatedTime'], format='%Y-%m-%dt%H:%M:%S.%f')
    X['DepartureScheduledTime'] = pd.to_datetime(X['DepartureScheduledTime'], format='%Y-%m-%dt%H:%M:%S.%f')
    X['DepartureEstimatedRunway'] = pd.to_datetime(X['DepartureEstimatedRunway'], format='%Y-%m-%dt%H:%M:%S.%f')
    X['DepartureActualRunway'] = pd.to_datetime(X['DepartureActualRunway'], format='%Y-%m-%dt%H:%M:%S.%f')

    X['Flight_Date'] = X['DepartureEstimatedTime'].dt.date
    X['DepartureEstimatedTime'] = X['DepartureEstimatedTime'].dt.time
    X['DepartureScheduledTime'] = X['DepartureScheduledTime'].dt.time
    X['DepartureEstimatedRunway'] = X['DepartureEstimatedRunway'].dt.time
    X['DepartureActualRunway'] = X['DepartureActualRunway'].dt.time


    categorical_columns = ['Status', 'DepartureAirport', 'DepartureTerminal', 'DepartureGate', 'ArrivalAirport', 'AirlineName']
    X_encoded = one_hot_encode_columns(X, categorical_columns)

    X_encoded.dropna(inplace=True)

    X_encoded['DepartureScheduledTimeHour'] = X_encoded['DepartureScheduledTime'].apply(lambda x: x.hour)
    X_encoded['DepartureScheduledTimeMinutes'] = X_encoded['DepartureScheduledTime'].apply(lambda x: x.minute)
    X_encoded['DepartureEstimatedTimeHour'] = X_encoded['DepartureEstimatedTime'].apply(lambda x: x.hour)
    X_encoded['DepartureEstimatedTimeMinutes'] = X_encoded['DepartureEstimatedTime'].apply(lambda x: x.minute)
    X_encoded['DepartureEstimatedRunwayHour'] = X_encoded['DepartureEstimatedRunway'].apply(lambda x: x.hour)
    X_encoded['DepartureEstimatedRunwayMinutes'] = X_encoded['DepartureEstimatedRunway'].apply(lambda x: x.minute)
    X_encoded['DepartureActualRunwayHour'] = X_encoded['DepartureActualRunway'].apply(lambda x: x.hour)
    X_encoded['DepartureActualRunwayMinutes'] = X_encoded['DepartureActualRunway'].apply(lambda x: x.minute)
    X_encoded['Flight_Date_Year'] = X_encoded['Flight_Date'].apply(lambda x: x.year)
    X_encoded['Flight_Date_Month'] = X_encoded['Flight_Date'].apply(lambda x: x.month)
    X_encoded['Flight_Date_Day'] = X_encoded['Flight_Date'].apply(lambda x: x.day)

    X_encoded.drop(['DepartureScheduledTime', 'DepartureEstimatedTime', 'DepartureEstimatedRunway', 'DepartureActualRunway', 'Flight_Date'], axis=1, inplace=True)

    return X_encoded, X

def train_and_evaluate_model(regressor, X_train, X_test, y_train, y_test):
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE): {:.2f}".format(mse))
    print("Mean Absolute Error (MAE): {:.2f}".format(mae))
    print("R-squared (R2): {:.2f}".format(r2))

    return regressor, mse, mae, r2

def make_predictions(regressor, X_test, y_test):
    # Make predictions
    y_pred = regressor.predict(X_test)

    # Display a scatter plot of predicted vs actual delays
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=y_test, ax=ax)
    ax.set_xlabel("Predicted Delays")
    ax.set_ylabel("Actual Delays")
    st.subheader("Predictions vs Actual Delays:")
    st.pyplot(fig)
    st.write("This scatter plot juxtaposes predicted departure delays against actual delays. Each point represents an instance from our test dataset. The closer points align to the diagonal line, the more accurate our model's predictions, demonstrating its effectiveness in forecasting flight delays.")

# Streamlit App
def main():
    st.set_page_config(
    page_title="Flight Delay Prediction",
    layout="wide",
    initial_sidebar_state="expanded",  # Expand the sidebar by default
)
    st.image('airportaerial.jpg')
    st.markdown("***")
    st.sidebar.title("Flight Delay Analysis & Prediction App")
    st.sidebar.markdown("***")
    st.sidebar.write(
            """
            ### Airport Codes:
            - :airplane: **DXB:** Dubai International Airport
            - :airplane: **LAS:** Harry Reid International Airport Los Angeles
            - :airplane: **ATL:** Hartsfield-Jackson Atlanta International Airport
            - :airplane: **DFW:** Dallas/Fort Worth International Airport
            - :airplane: **ORD:** O'Hare International Airport
            """
        )
    selected_page = st.sidebar.radio("Select Page", ["Analysis", "Prediction"])
    if selected_page == "Analysis":

        col1, col2 = st.columns(2)
        st.sidebar.write("***Please note:*** the API data has a three-day delay. Dates within the last three days are not available. If you're interested in predictive analysis for future dates, kindly switch to the Predictive Plots page using the options above.")
        st.sidebar.subheader("Input Controls:")
        selected_airports = st.sidebar.multiselect("Select Airports (among world's top 5 airports):", ['dxb', 'las', 'atl', 'dfw', 'ord'], default=['dxb'])

        # Select date range using sliders
        with col1:
            start_date = st.sidebar.date_input("Select Start Date:", pd.to_datetime('2023-11-01'))
        with col2:
            end_date = st.sidebar.date_input("Select End Date:", pd.to_datetime('2023-11-03'))
        
        num_airports_input = len(selected_airports)
        date_range_input = (end_date - start_date).days
        estimated_time = estimate_processing_time_quadratic(num_airports_input, date_range_input)
        st.sidebar.write(f"Estimated time to generate and display results: {estimated_time:.2f} seconds")

        hit_me_button = st.sidebar.button('Run Analysis')

        if hit_me_button:
            with st.spinner('Pulling data from API...'):
                # Load and preprocess the data
                df, df_X = load_and_preprocess_data(selected_airports, start_date, end_date)
                # Display a sample of the dataset
                st.subheader("Sample of the Dataset:")
                st.dataframe(df_X.head())

            with st.spinner('Loading analysis...'):
                # Train and evaluate the model
                X, y = df.drop('DepartureDelay', axis=1), df['DepartureDelay']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                rf = RandomForestRegressor()
                rf_trained_model, mse, mae, r2 = train_and_evaluate_model(rf, X_train, X_test, y_train, y_test)

                # Display model evaluation metrics
                st.subheader("Model Evaluation Metrics:")
                st.write("Mean Squared Error (MSE): {:.2f}".format(mse))
                st.write("Mean Absolute Error (MAE): {:.2f}".format(mae))
                st.write("R-squared (R2): {:.2f}".format(r2))

                # Make predictions and display results
                # st.subheader("Predictions:")
                make_predictions(rf_trained_model, X_test, y_test)

                mean_delays = df_X.groupby(['Flight_Date', 'DepartureAirport'])['DepartureDelay'].median().reset_index()

                # Plot the median delays using Plotly Express
                fig = px.line(mean_delays, x='Flight_Date', y='DepartureDelay', color='DepartureAirport')
                fig.update_xaxes(title_text='Flight Date')
                fig.update_yaxes(title_text='Median Departure Delay (minutes)')

                st.subheader("Median Departure Delays Over Time for Each Airport:")
                st.plotly_chart(fig)
                st.write('This line chart illustrates the median departure delays over time for each airport. The x-axis represents the flight date, while the y-axis indicates the median departure delay in minutes. Each line corresponds to a different airport, providing a comprehensive view of how median delays fluctuate across various dates and airports.')
                st.write("In the context of departure delays, the median is often preferred over the mean due to its robustness against extreme values or outliers. Departure delay data may occasionally contain outliers, such as unusually long delays caused by exceptional circumstances. These outliers can significantly impact the mean, making it less representative of the typical delay experience.")

                mean_delays = df_X.groupby(['Flight_Date', 'DepartureAirport'])['DepartureDelay'].max().reset_index()

                # Plot the maximum delays using Plotly Express
                fig = px.line(mean_delays, x='Flight_Date', y='DepartureDelay', color='DepartureAirport')
                fig.update_xaxes(title_text='Flight Date')
                fig.update_yaxes(title_text='Maximum Departure Delay (minutes)')

                st.subheader("Maximum Departure Delays Over Time for Each Airport:")
                st.plotly_chart(fig)
                st.write('The plot visualizes the maximum departure delays for each airport on different dates. Unlike measures such as the median or mean, which provide insights into typical delay experiences, the maximum delay sheds light on the extreme situations where flights encounter unusually long delays. No one likes a 5-hour wait for their flight!')

    if selected_page == 'Prediction':
        st.sidebar.subheader("Input Controls for Prediction:")
        selected_airport = st.sidebar.multiselect("Select One Airport:", ['dxb', 'las', 'atl', 'dfw', 'ord'])
        prediction_date = st.sidebar.date_input("Select Prediction Date: (if it's a past date, you get a comparison analysis!)", pd.to_datetime('2023-11-10'))
        hit_me_button2 = st.sidebar.button('Run Prediction')
        if hit_me_button2:
            with st.spinner('Running prediction...'):
                current_datetime = datetime.now()
                
                if prediction_date <= current_datetime.date():
                    df, df_X = load_and_preprocess_data(selected_airport, prediction_date, prediction_date)
                    rf = RandomForestRegressor()
                    X, y = df.drop('DepartureDelay', axis=1), df['DepartureDelay']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                    rf_trained_model, mse, mae, r2 = train_and_evaluate_model(rf, X_train, X_test, y_train, y_test)
                    predictions = rf_trained_model.predict(df.drop('DepartureDelay', axis=1))

                    for i in selected_airport:
                        st.subheader(f'Predicted Delay for {i} on {prediction_date}:')
                        st.write(f'{predictions.mean():.0f} minutes')

                    # Get the actual delays for the specified date
                    actual_delays = df_X[df_X['Flight_Date'] == prediction_date]['DepartureDelay'].tolist()

                    # Display the ground truth delays
                    st.subheader(f'Actual Delays for {i} on {prediction_date}:')
                    median_actual_delay = np.nanmedian(actual_delays)
                    st.write(f'Ground Truth Delays: {median_actual_delay}')

                    st.subheader('Comparison:')
                    st.write(f'Predicted Delay: {predictions.mean():.0f} minutes')
                    st.write(f'Actual Delay: {median_actual_delay:.0f} minutes')

                    # Plotting the comparison
                    pastel_colors = ['#FFB6C1', '#87CEFA']
                    labels = ['Predicted Delay', 'Actual Delay']
                    values = [round(predictions.mean(),0), median_actual_delay]

                    fig, ax = plt.subplots()
                    bars = ax.bar(labels, values, color=pastel_colors)

                    # Adding labels and title
                    ax.set_ylabel('Delay (minutes)')
                    ax.set_title('Comparison of Predicted and Actual Delays')

                    # Adding text labels on top of the bars
                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

                    # Display the plot in Streamlit
                    st.pyplot(fig)

                else:
                    four_days_ago = current_datetime - timedelta(days=4)
                    five_days_ago = current_datetime - timedelta(days=5)
                    df, df_X = load_and_preprocess_data(selected_airport, five_days_ago, four_days_ago)
                    rf = RandomForestRegressor()
                    X, y = df.drop('DepartureDelay', axis=1), df['DepartureDelay']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                    rf_trained_model, mse, mae, r2 = train_and_evaluate_model(rf, X_train, X_test, y_train, y_test)
                    predictions = rf_trained_model.predict(df.drop('DepartureDelay', axis=1))

                    for i in selected_airport:
                        st.subheader(f'Predicted Delay for {i} on {prediction_date}:')
                        st.write(f'{predictions.mean():.0f} minutes')

if __name__ == '__main__':
    main()