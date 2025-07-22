import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from duckduckgo_search import DDGS

# Load environment
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

# Initialize Groq model
model = ChatGroq(model_name="llama3-70b-8192", api_key=api_key)

# DuckDuckGo search function
def search_duckduckgo(query, num_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append(f"{r['title']}: {r['body']} ({r['href']})")
    return "\n\n".join(results)

# Output parser
parser = StrOutputParser()

# Streamlit UI
st.title("🕵️ Fake News Detection Model (with Internet Search)")
st.header("Paste any news below, the model will check if it's Fake or Not using real-time data.")

user_input = st.text_area("Enter your news statement", height=200)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a news statement.")
    else:
        with st.spinner("Searching the web and analyzing..."):
            context = search_duckduckgo(user_input)

            if context:
                st.success("✅ Model is connected to live internet search via DuckDuckGo")
                st.markdown("#### 🔍 Search Context:")
                st.write(context)
            else:
                st.error("❌ Failed to retrieve search results. Model is not internet-connected.")

            # Prompt template with real-time search context
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a highly accurate fake news detection expert. Use the provided search results to decide if the claim is Fake or Not Fake.

Output Format:
- Verdict: Fake / Not Fake
- Confidence: (High / Medium / Low)
- Explanation: (Brief reasoning or fact-check)
- Source Links (if used): [List of URLs or 'Not Available']"""),
                ("user", f"""Claim: {user_input}

Search Results:
{context}

Please analyze the above and respond in the specified format.""")
            ])

            chain = prompt | model | parser
            response = chain.invoke({})

            st.subheader("📄 Model Verdict")
            st.markdown(response)
