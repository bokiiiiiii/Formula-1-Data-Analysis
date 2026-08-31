import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

username = os.environ.get("INSTAGRAM_USERNAME")
password = os.environ.get("INSTAGRAM_PASSWORD")

print(f"Username: {username}")
print(f"Password: {password}")
