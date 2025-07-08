import requests

# The text you want to send for prediction
text_input_1 = "The latest news about the upcoming elections in the country."
text_input_2 = "Investors are closely watching the stock market trends this week."
text_input_3 = "The national team secured a decisive victory in the championship."

# Define the headers, explicitly setting Content-Type to text/plain
headers = {"Content-Type": "text/plain"}

def test_prediction(text):
    print(f"\n--- Testing with input: '{text}' ---")
    try:
        # Send the POST request to the BentoML service on port 10000
        response = requests.post(
            "http://localhost:10000/predict",
            data=text.encode('utf-8'), # Encode the string to bytes for plain text
            headers=headers
        )

        # Check the response
        if response.status_code == 200:
            print(f"Prediction: {response.text}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response Body: {response.text}") # Print the full response body for debugging

    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Could not connect to the BentoML service at http://localhost:10000.")
        print(f"Please ensure the 'bentoml serve' command is running in a separate terminal.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run tests
test_prediction(text_input_1)
test_prediction(text_input_2)
test_prediction(text_input_3)