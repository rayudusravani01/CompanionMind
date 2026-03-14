import pickle

# Path to the user database file
USER_DB = "user_db1.pkl"

try:
    # Load the user database
    with open(USER_DB, "rb") as f:
        users = pickle.load(f)
    
    # Display the user data
    print("User Database:")
    if isinstance(users, list):  # Ensure it's a list of dictionaries
        for user in users:
            print(
                f"First Name: {user.get('first_name', 'N/A')}, "
                f"Last Name: {user.get('last_name', 'N/A')}, "
                f"Phone Number: {user.get('phone_number', 'N/A')}, "
                f"Username: {user.get('username', 'N/A')}, "
                f"Hashed Password: {user.get('password', 'N/A')}"
            )
    else:
        print("The user database is not in the expected format.")
except FileNotFoundError:
    print(f"The file '{USER_DB}' does not exist.")
except Exception as e:
    print(f"An error occurred while reading the user database: {e}")
