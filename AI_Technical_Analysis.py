# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go
# import ollama
# import tempfile
# import base64
# import os

# # Set up Streamlit app
# st.set_page_config(layout="wide")
# st.title("AI-Powered Technical Stock Analysis Dashboard")
# st.sidebar.header("Configuration")

# # Input for stock ticker and date range
# ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
# start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
# end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-14"))

# # Fetch stock data
# if st.sidebar.button("Fetch Data"):
#     st.session_state["stock_data"] = yf.download(ticker, start=start_date, end=end_date)
#     st.success("Stock data loaded successfully!")

# # Check if data is available
# if "stock_data" in st.session_state:
#     data = st.session_state["stock_data"]

#     # Plot candlestick chart
#     fig = go.Figure(data=[
#         go.Candlestick(
#             x=data.index,
#             open=data['Open'],
#             high=data['High'],
#             low=data['Low'],
#             close=data['Close'],
#             name="Candlestick"  # Replace "trace 0" with "Candlestick"
#         )
#     ])

#     # Sidebar: Select technical indicators
#     st.sidebar.subheader("Technical Indicators")
#     indicators = st.sidebar.multiselect(
#         "Select Indicators:",
#         ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
#         default=["20-Day SMA"]
#     )

#     # Helper function to add indicators to the chart
#     def add_indicator(indicator):
#         if indicator == "20-Day SMA":
#             sma = data['Close'].rolling(window=20).mean()
#             fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
#         elif indicator == "20-Day EMA":
#             ema = data['Close'].ewm(span=20).mean()
#             fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
#         elif indicator == "20-Day Bollinger Bands":
#             sma = data['Close'].rolling(window=20).mean()
#             std = data['Close'].rolling(window=20).std()
#             bb_upper = sma + 2 * std
#             bb_lower = sma - 2 * std
#             fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
#             fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
#         elif indicator == "VWAP":
#             data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
#             fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

#     # Add selected indicators to the chart
#     for indicator in indicators:
#         add_indicator(indicator)

#     fig.update_layout(xaxis_rangeslider_visible=False)
#     st.plotly_chart(fig)

#     # Analyze chart with LLaMA 3.2 Vision
#     st.subheader("AI-Powered Analysis")
#     if st.button("Run AI Analysis"):
#         with st.spinner("Analyzing the chart, please wait..."):
#             # Save chart as a temporary image
#             with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
#                 fig.write_image(tmpfile.name)
#                 tmpfile_path = tmpfile.name

#             # Read image and encode to Base64
#             with open(tmpfile_path, "rb") as image_file:
#                 image_data = base64.b64encode(image_file.read()).decode('utf-8')

#             # Prepare AI analysis request
#             messages = [{
#                 'role': 'user',
#                 'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
#                             Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
#                             Base your recommendation only on the candlestick chart and the displayed technical indicators.
#                             First, provide the recommendation, then, provide your detailed reasoning.
#                 """,
#                 'images': [image_data]
#             }]
#             response = ollama.chat(model='llama3.2-vision', messages=messages)

#             # Display AI analysis result
#             st.write("**AI Analysis Results:**")
#             st.write(response["message"]["content"])

#             # Clean up temporary file
#             os.remove(tmpfile_path)








# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go
# import ollama
# import tempfile
# import base64
# import os

# # Set up Streamlit app
# st.set_page_config(layout="wide")
# st.title("AI-Powered Technical Stock Analysis Dashboard")
# st.sidebar.header("Configuration")

# # Input for stock ticker and date range
# ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
# start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
# end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-14"))

# # Fetch stock data
# if st.sidebar.button("Fetch Data"):
#     stock_data = yf.download(ticker, start=start_date, end=end_date)
#     if not stock_data.empty:
#         st.session_state["stock_data"] = stock_data
#         st.success("Stock data loaded successfully!")
#     else:
#         st.error("Failed to load stock data. Please check the ticker or date range.")

# # Check if data is available
# if "stock_data" in st.session_state:
#     data = st.session_state["stock_data"]

#     if data.empty:
#         st.warning("No data available for the selected ticker and date range.")
#     else:
#         # Plot candlestick chart
#         fig = go.Figure(data=[
#             go.Candlestick(
#                 x=data.index,
#                 open=data['Open'],
#                 high=data['High'],
#                 low=data['Low'],
#                 close=data['Close'],
#                 name="Candlestick"
#             )
#         ])

#         # Sidebar: Select technical indicators
#         st.sidebar.subheader("Technical Indicators")
#         indicators = st.sidebar.multiselect(
#             "Select Indicators:",
#             ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
#             default=["20-Day SMA"]
#         )

#         # Helper function to add indicators to the chart
#         def add_indicator(indicator):
#             if indicator == "20-Day SMA":
#                 sma = data['Close'].rolling(window=20).mean()
#                 fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
#             elif indicator == "20-Day EMA":
#                 ema = data['Close'].ewm(span=20).mean()
#                 fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
#             elif indicator == "20-Day Bollinger Bands":
#                 sma = data['Close'].rolling(window=20).mean()
#                 std = data['Close'].rolling(window=20).std()
#                 bb_upper = sma + 2 * std
#                 bb_lower = sma - 2 * std
#                 fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
#                 fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
#             elif indicator == "VWAP":
#                 data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
#                 fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

#         # Add selected indicators to the chart
#         for indicator in indicators:
#             add_indicator(indicator)

#         fig.update_layout(xaxis_rangeslider_visible=False)
#         st.plotly_chart(fig)

#         # Analyze chart with LLaMA 3.2 Vision
#         st.subheader("AI-Powered Analysis")
#         if st.button("Run AI Analysis"):
#             with st.spinner("Analyzing the chart, please wait..."):
#                 # Save chart as a temporary image
#                 with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
#                     fig.write_image(tmpfile.name)
#                     tmpfile_path = tmpfile.name

#                 # Read image and encode to Base64
#                 with open(tmpfile_path, "rb") as image_file:
#                     image_data = base64.b64encode(image_file.read()).decode('utf-8')

#                 # Prepare AI analysis request
#                 messages = [{
#                     'role': 'user',
#                     'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
#                                 Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
#                                 Base your recommendation only on the candlestick chart and the displayed technical indicators.
#                                 First, provide the recommendation, then, provide your detailed reasoning.
#                     """,
#                     'images': [image_data]
#                 }]
#                 response = ollama.chat(model='llama3.2-vision', messages=messages)

#                 # Display AI analysis result
#                 st.write("**AI Analysis Results:**")
#                 st.write(response["message"]["content"])

#                 # Clean up temporary file
#                 os.remove(tmpfile_path)








import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-14"))

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    try:
        # Fetch data using yfinance
        raw_data = yf.download(ticker, start=start_date, end=end_date)

        # Check the raw fetched data structure
        st.write("Raw fetched data:", raw_data.head())
        st.write("Raw fetched data columns:", raw_data.columns)

        if raw_data.empty:
            st.error("No data fetched. Please check the ticker symbol or date range.")
        else:
            # Handle the column format to extract the relevant data
            # The column names seem to include the stock ticker (e.g., AMZN)
            # We will extract the 'AMZN' (or the relevant ticker) part from each column
            # First, ensure the columns are properly formatted (e.g., 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume')
            raw_data.columns = raw_data.columns.droplevel(1)  # Drop the 'AMZN' level

            # Now that we have cleaned the columns, check for the expected columns
            column_names = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            
            # Check if all the necessary columns are present
            missing_columns = [col for col in column_names if col not in raw_data.columns]
            if missing_columns:
                st.warning(f"Missing columns: {', '.join(missing_columns)}")

            # Filter only the relevant columns
            data_cleaned = raw_data[column_names]

            # Reset the index to make sure the data is in the correct format
            data_cleaned.reset_index(inplace=True)
            data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])  # Ensure 'Date' is a datetime object
            data_cleaned.set_index('Date', inplace=True) 

            # Clean the column names
            data_cleaned.columns = [col.strip() for col in data_cleaned.columns]  # Clean column names

            # Update the session state with cleaned data
            st.session_state["stock_data"] = data_cleaned
            st.success("Stock data loaded successfully!")

    except Exception as e:
        st.error(f"Error fetching data: {e}")


# Check if data is available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    if data.empty:
        st.warning("No data available for the selected ticker and date range.")
    else:
        # Display raw data
        st.write("Raw Data Preview")
        st.write(data.head())

        # Sidebar: Select technical indicators
        st.sidebar.subheader("Technical Indicators")
        indicators = st.sidebar.multiselect(
            "Select Indicators:",
            ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
            default=["20-Day SMA"]
        )

        # Plot candlestick chart
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            )
        )

        # Helper function to add indicators to the chart
        def add_indicator(indicator):
            if indicator == "20-Day SMA":
                sma = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
            elif indicator == "20-Day EMA":
                ema = data['Close'].ewm(span=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
            elif indicator == "20-Day Bollinger Bands":
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            elif indicator == "VWAP":
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

        # Add selected indicators to the chart
        for indicator in indicators:
            add_indicator(indicator)

        # Customize layout and display chart
        fig.update_layout(
            title=f"{ticker} Stock Price with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Analyze chart with AI
        st.subheader("AI-Powered Analysis")
        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing the chart, please wait..."):
                # Save chart as a temporary image
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.write_image(tmpfile.name)
                    tmpfile_path = tmpfile.name

                # Read image and encode to Base64
                with open(tmpfile_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')

                # Prepare AI analysis request
                messages = [{
                    'role': 'user',
                    'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                                Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                                Base your recommendation only on the candlestick chart and the displayed technical indicators.
                                First, provide the recommendation, then, provide your detailed reasoning.
                    """,
                    'images': [image_data]
                }]
                response = ollama.chat(model='llama3.2-vision', messages=messages)

                # Display AI analysis result
                st.write("**AI Analysis Results:**")
                st.write(response["message"]["content"])

                # Clean up temporary file
                os.remove(tmpfile_path)







































# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import mplfinance as mpf
# import ollama
# import base64
# import os
# from io import BytesIO

# # Set up Streamlit app
# st.set_page_config(layout="wide")
# st.title("AI-Powered Technical Stock Analysis Dashboard")
# st.sidebar.header("Configuration")

# # Input for stock ticker and date range
# ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
# start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
# end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-14"))

# # Fetch stock data
# # Fetch stock data
# # Fetch stock data
# # Fetch stock data
# # Fetch stock data
# # Fetch stock data
# if st.sidebar.button("Fetch Data"):
#     try:
#         # Fetch data using yfinance
#         raw_data = yf.download(ticker, start=start_date, end=end_date)

#         # Check the raw fetched data structure
#         st.write("Raw fetched data:", raw_data.head())
#         st.write("Raw fetched data columns:", raw_data.columns)

#         if raw_data.empty:
#             st.error("No data fetched. Please check the ticker symbol or date range.")
#         else:
#             # Handle the column format to extract the relevant data
#             # The column names seem to include the stock ticker (e.g., AMZN)
#             # We will extract the 'AMZN' (or the relevant ticker) part from each column
#             # First, ensure the columns are properly formatted (e.g., 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume')
#             raw_data.columns = raw_data.columns.droplevel(1)  # Drop the 'AMZN' level

#             # Now that we have cleaned the columns, check for the expected columns
#             column_names = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            
#             # Check if all the necessary columns are present
#             missing_columns = [col for col in column_names if col not in raw_data.columns]
#             if missing_columns:
#                 st.warning(f"Missing columns: {', '.join(missing_columns)}")

#             # Filter only the relevant columns
#             data_cleaned = raw_data[column_names]

#             # Reset the index to make sure the data is in the correct format
#             data_cleaned.reset_index(inplace=True)
#             data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])  # Ensure 'Date' is a datetime object
#             data_cleaned.set_index('Date', inplace=True) 

#             # Clean the column names
#             data_cleaned.columns = [col.strip() for col in data_cleaned.columns]  # Clean column names

#             # Update the session state with cleaned data
#             st.session_state["stock_data"] = data_cleaned
#             st.success("Stock data loaded successfully!")

#     except Exception as e:
#         st.error(f"Error fetching data: {e}")



# # # Check if data is available
# # if "stock_data" in st.session_state:
# #     data = st.session_state["stock_data"]

# #     # Drop rows with missing values in critical columns
# #     critical_columns = ["Open", "High", "Low", "Close", "Volume"]
# #     data = data.dropna(subset=critical_columns)

# #     # Convert critical columns to numeric types
# #     for column in critical_columns:
# #         data[column] = pd.to_numeric(data[column], errors='coerce')

# #     # Drop any rows with remaining NaNs after conversion
# #     data = data.dropna()

# #     # Sidebar: Select technical indicators
# #     st.sidebar.subheader("Technical Indicators")
# #     indicators = st.sidebar.multiselect(
# #         "Select Indicators:",
# #         ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands"],
# #         default=["20-Day SMA"]
# #     )

# #     # Calculate technical indicators
# #     if "20-Day SMA" in indicators:
# #         data["SMA_20"] = data["Close"].rolling(window=20).mean()
# #     if "20-Day EMA" in indicators:
# #         data["EMA_20"] = data["Close"].ewm(span=20).mean()
# #     if "20-Day Bollinger Bands" in indicators:
# #         data["BB_MID"] = data["Close"].rolling(window=20).mean()
# #         data["BB_UPPER"] = data["BB_MID"] + 2 * data["Close"].rolling(window=20).std()
# #         data["BB_LOWER"] = data["BB_MID"] - 2 * data["Close"].rolling(window=20).std()

# #     # Prepare mplfinance plot settings
# #     additional_plots = []
# #     if "20-Day SMA" in indicators:
# #         additional_plots.append(mpf.make_addplot(data["SMA_20"], color='blue', label='SMA 20'))
# #     if "20-Day EMA" in indicators:
# #         additional_plots.append(mpf.make_addplot(data["EMA_20"], color='orange', label='EMA 20'))
# #     if "20-Day Bollinger Bands" in indicators:
# #         additional_plots.append(mpf.make_addplot(data["BB_UPPER"], color='green', label='BB Upper'))
# #         additional_plots.append(mpf.make_addplot(data["BB_LOWER"], color='red', label='BB Lower'))

# #     # Create the mplfinance candlestick chart
# #     fig, ax = mpf.plot(
# #         data,
# #         type='candle',
# #         addplot=additional_plots,
# #         title=f"{ticker} Stock Price",
# #         ylabel='Price',
# #         volume=True,
# #         returnfig=True
# #     )

# #     # Save the chart as an image buffer
# #     img_buffer = BytesIO()
# #     fig.savefig(img_buffer, format='png')
# #     img_buffer.seek(0)

# #     # Display the mplfinance chart in Streamlit
# #     st.image(img_buffer, caption=f"{ticker} Candlestick Chart with Indicators", use_column_width=True)
# #     img_buffer.close()

# #     # Analyze chart with AI (LLaMA Vision Model)
# #     st.subheader("AI-Powered Analysis")
# #     if st.button("Run AI Analysis"):
# #         with st.spinner("Analyzing the chart, please wait..."):
# #             # Encode image to Base64
# #             img_buffer = BytesIO()
# #             fig.savefig(img_buffer, format='png')
# #             img_buffer.seek(0)
# #             image_data = base64.b64encode(img_buffer.read()).decode('utf-8')
# #             img_buffer.close()

# #             # Prepare AI analysis request
# #             messages = [{
# #                 'role': 'user',
# #                 'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
# #                             Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
# #                             Base your recommendation only on the candlestick chart and the displayed technical indicators.
# #                             First, provide the recommendation, then, provide your detailed reasoning.
# #                 """,
# #                 'images': [image_data]
# #             }]
# #             response = ollama.chat(model='llama3.2-vision', messages=messages)

# #             # Display AI analysis result
# #             st.write("**AI Analysis Results:**")
# #             st.write(response["message"]["content"])

# # Check if data is available
# if "stock_data" in st.session_state:
#     data = st.session_state["stock_data"]

#     st.write("Columns in the fetched data:", data.columns.tolist())

#     # Check if the required columns exist in the data
#     required_columns = ["Open", "High", "Low", "Close", "Volume"]
#     if not all(col in data.columns for col in required_columns):
#         st.error(f"Missing required columns in the data: {', '.join([col for col in required_columns if col not in data.columns])}")
#     else:
#         # Drop rows with missing values in critical columns
#         data = data.dropna(subset=required_columns)

#         # Convert critical columns to numeric types
#         for column in required_columns:
#             data[column] = pd.to_numeric(data[column], errors='coerce')

#         # Drop any rows with remaining NaNs after conversion
#         data = data.dropna()

#         # Sidebar: Select technical indicators
#         st.sidebar.subheader("Technical Indicators")
#         indicators = st.sidebar.multiselect(
#             "Select Indicators:",
#             ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands"],
#             default=["20-Day SMA"]
#         )

#         # Calculate technical indicators
#         if "20-Day SMA" in indicators:
#             data["SMA_20"] = data["Close"].rolling(window=20).mean()
#         if "20-Day EMA" in indicators:
#             data["EMA_20"] = data["Close"].ewm(span=20).mean()
#         if "20-Day Bollinger Bands" in indicators:
#             data["BB_MID"] = data["Close"].rolling(window=20).mean()
#             data["BB_UPPER"] = data["BB_MID"] + 2 * data["Close"].rolling(window=20).std()
#             data["BB_LOWER"] = data["BB_MID"] - 2 * data["Close"].rolling(window=20).std()

#         # Prepare mplfinance plot settings
#         additional_plots = []
#         if "20-Day SMA" in indicators:
#             additional_plots.append(mpf.make_addplot(data["SMA_20"], color='blue', label='SMA 20'))
#         if "20-Day EMA" in indicators:
#             additional_plots.append(mpf.make_addplot(data["EMA_20"], color='orange', label='EMA 20'))
#         if "20-Day Bollinger Bands" in indicators:
#             additional_plots.append(mpf.make_addplot(data["BB_UPPER"], color='green', label='BB Upper'))
#             additional_plots.append(mpf.make_addplot(data["BB_LOWER"], color='red', label='BB Lower'))

#         # Create the mplfinance candlestick chart
#         fig, ax = mpf.plot(
#             data,
#             type='candle',
#             addplot=additional_plots,
#             title=f"{ticker} Stock Price",
#             ylabel='Price',
#             volume=True,
#             returnfig=True
#         )

#         # Save the chart as an image buffer
#         img_buffer = BytesIO()
#         fig.savefig(img_buffer, format='png')
#         img_buffer.seek(0)

#         # Display the mplfinance chart in Streamlit
#         st.image(img_buffer, caption=f"{ticker} Candlestick Chart with Indicators", use_column_width=True)
#         img_buffer.close()

#         # Analyze chart with AI (LLaMA Vision Model)
#         st.subheader("AI-Powered Analysis")
#         if st.button("Run AI Analysis"):
#             with st.spinner("Analyzing the chart, please wait..."):
#                 # Encode image to Base64
#                 img_buffer = BytesIO()
#                 fig.savefig(img_buffer, format='png')
#                 img_buffer.seek(0)
#                 image_data = base64.b64encode(img_buffer.read()).decode('utf-8')
#                 img_buffer.close()

#                 # Prepare AI analysis request
#                 messages = [{
#                     'role': 'user',
#                     'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
#                                 Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
#                                 Base your recommendation only on the candlestick chart and the displayed technical indicators.
#                                 First, provide the recommendation, then, provide your detailed reasoning.
#                     """,
#                     'images': [image_data]
#                 }]
#                 response = ollama.chat(model='llama3.2-vision', messages=messages)

#                 # Display AI analysis result
#                 st.write("**AI Analysis Results:**")
#                 st.write(response["message"]["content"])
