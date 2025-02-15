import streamlit as st
from authlib.integrations.requests_client import OAuth2Session
import os

# Configuration
CLIENT_ID = "your-google-client-id"
CLIENT_SECRET = "your-google-client-secret"
AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REDIRECT_URI = "http://localhost:8501"
SCOPES = ["openid", "email", "profile"]

def get_google_auth_session():
    return OAuth2Session(CLIENT_ID, CLIENT_SECRET, scope=SCOPES, redirect_uri=REDIRECT_URI)

def login():
    client = get_google_auth_session()
    authorization_url, state = client.create_authorization_url(AUTHORIZATION_URL)
    st.session_state["oauth_state"] = state
    st.markdown(f"[Login with Google]({authorization_url})")

def fetch_token(auth_response):
    client = get_google_auth_session()
    token = client.fetch_token(TOKEN_URL, authorization_response=auth_response, client_secret=CLIENT_SECRET)
    return token

def main():
    st.title("Login with Google")
    if "oauth_token" not in st.session_state:
        login()
    else:
        st.success("You are logged in!")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
