#Database@1301200 Supabase database password
from os import getenv
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    getenv("SUPABASE_URL"),
    getenv("SUPABASE_KEY")
)

