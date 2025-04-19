import streamlit as st
from supabase import create_client, Client
import pandas as pd
from collections import Counter
import google.generativeai as genai
import altair as alt


# --- Supabase credentials ---
# --- Supabase Setup ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("📊 Grab Driver Voice Assistant - Chat Log Dashboard")

# --- Fetch data from Supabase ---
# --- Fetch data from Supabase ---
with st.spinner("Fetching chat logs from Supabase..."):
    response = supabase.table("driver_logs").select("chat_log, created_at").execute()

    logs = response.data

    if not logs:
        st.warning("⚠️ No chat logs found.")
    else:
        structured_rows = []

        for row in logs:
            chat_log = row.get("chat_log", {})
            messages = chat_log.get("messages", [])
            created_at = row.get("created_at")

            if isinstance(messages, list) and len(messages) >= 3:
                # Skip first assistant message (greeting)
                i = 1
                while i < len(messages) - 1:
                    user_msg = messages[i]
                    assistant_msg = messages[i + 1]

                    # Ensure it's a valid user-assistant pair
                    if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                        structured_rows.append({
                            "user_message": user_msg.get("content", ""),
                            "intent": user_msg.get("intent", ""),
                            "assistant_response": assistant_msg.get("content", ""),
                            "created_at": created_at
                        })
                        i += 2  # Skip to next user-assistant pair
                    else:
                        i += 1  # Just in case roles mismatch, move ahead

        if not structured_rows:
            st.warning("⚠️ No structured user-assistant message pairs found.")
        else:
            df = pd.DataFrame(structured_rows)

            with st.expander("🗂 Structured Chat Logs (User Request, Intent, Assistant Response)"):
                st.dataframe(df)

                # Export to CSV
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Structured Chat Logs as CSV",
                    data=csv_data,
                    file_name="structured_chat_logs.csv",
                    mime="text/csv"
                )

        # --- Analytics Section ---
        # st.subheader("📈 Basic Chat Analytics")

        # # 1. Message counts per role
        # role_counts = Counter([msg["role"] for msg in all_chats])
        # st.markdown("**Message Count by Role:**")
        # st.bar_chart(pd.DataFrame.from_dict(role_counts, orient='index', columns=["Count"]))
        
        # # --- Interaction Success Rate ---
        st.markdown("### ✅ Interaction Success Analysis")

        # Filter only user messages with 'intent' field
        intent_msgs = [
            {"role": "user", "intent": row["intent"]} 
            for _, row in df.iterrows() 
            if row["intent"] and row["intent"] != ""
        ]

        if not intent_msgs:
            st.info("No intents found in user messages to evaluate success rate.")
        else:
            total_intents = len(intent_msgs)
            successful = sum(1 for msg in intent_msgs if msg["intent"] != "unknown")
            unsuccessful = total_intents - successful

            success_percent = (successful / total_intents) * 100
            fail_percent = (unsuccessful / total_intents) * 100

            cols = st.columns(2)

            with cols[0]:
                st.markdown(f"""
                    <div style='
                        border: 1px solid #444;
                        border-radius: 10px;
                        padding: 1.2rem;
                        background-color: rgba(255, 255, 255, 0.02);
                        text-align: center;
                    '>
                        <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem;'>
                            Successful Interactions
                        </div>
                        <div style='font-size: 2rem; font-weight: bold; margin: 0;'>{success_percent:.1f}%</div>
                        <div style='color: limegreen; margin-top: 0.3rem;'> {successful} of {total_intents} messages</div>
                    </div>
                """, unsafe_allow_html=True)

            with cols[1]:
                st.markdown(f"""
                    <div style='
                        border: 1px solid #444;
                        border-radius: 10px;
                        padding: 1.2rem;
                        background-color: rgba(255, 255, 255, 0.02);
                        text-align: center;
                    '>
                        <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem;'>
                            Unsuccessful Interactions
                        </div>
                        <div style='font-size: 2rem; font-weight: bold; margin: 0;'>{fail_percent:.1f}%</div>
                        <div style='color: orangered; margin-top: 0.3rem;'> {unsuccessful} of {total_intents} messages</div>
                    </div>
                """, unsafe_allow_html=True)



        # 2. Keyword Category Detection (Stylized)
        
        st.markdown("<div style='margin-top: 1.5rem'></div>", unsafe_allow_html=True)

        
        # Get all user messages with known intents (excluding 'unknown')
        known_intent_df = df[df["intent"] != "unknown"]
        intent_counts = known_intent_df["intent"].value_counts().to_dict()
        total_intent_msgs = sum(intent_counts.values())

        # Display top 3 intents (or all if fewer)
        max_intents_to_display = 3
        top_intents = dict(sorted(intent_counts.items(), key=lambda item: item[1], reverse=True)[:max_intents_to_display])

        st.markdown("### 🎯 Top User Intents")

        cols = st.columns(len(top_intents))
        for idx, (intent_name, count) in enumerate(top_intents.items()):
            percent = (count / total_intent_msgs) * 100 if total_intent_msgs > 0 else 0

            with cols[idx]:
                st.markdown(f"""
                    <div style='
                        border: 1px solid #444;
                        border-radius: 10px;
                        padding: 1.2rem;
                        background-color: rgba(255, 255, 255, 0.02);
                        text-align: center;
                    '>
                        <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.4rem;'>
                            <span style="text-transform: capitalize;">{intent_name}</span>
                        </div>
                        <div style='font-size: 2rem; font-weight: bold; margin: 0;'>{percent:.1f}%</div>
                        <div style='color: limegreen; margin-top: 0.3rem;'>↑ {count} message{"s" if count != 1 else ""}</div>
                    </div>
                """, unsafe_allow_html=True)



                # st.markdown(f"<h4 style='margin-bottom: 0.5rem'>{category}</h4>", unsafe_allow_html=True)
                # st.markdown(f"<h2 style='margin: 0'>{percent:.1f}%</h2>", unsafe_allow_html=True)
                # st.markdown(f"<span style='color: limegreen'>↑ {count} messages</span>", unsafe_allow_html=True)

                # st.markdown("</div>", unsafe_allow_html=True)
                
        # st.markdown("### 🔤 Most Common Words (User Messages)")
        
        # user_messages = df[df["role"] == "user"]["content"].dropna().tolist()
        # word_counter = Counter()

        # for msg in user_messages:
        #     words = msg.lower().split()
        #     cleaned_words = [w.strip(".,!?\"'()[]") for w in words if len(w) > 2]
        #     word_counter.update(cleaned_words)

        # # Convert to DataFrame and show top N
        # common_words_df = pd.DataFrame(word_counter.most_common(20), columns=["Word", "Frequency"])

        # st.table(common_words_df)
        
        st.markdown(" ")
        st.markdown(" ")
        
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["created_at"] = df["created_at"].dt.tz_convert('Asia/Kuala_Lumpur')

        # Time-of-day function
        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return "Morning ☀️"
            elif 12 <= hour < 17:
                return "Afternoon 🌤️"
            elif 17 <= hour < 21:
                return "Evening 🌇"
            else:
                return "Night 🌙"

        # Add time_of_day column
        df["time_of_day"] = df["created_at"].dt.hour.apply(get_time_of_day)

        # Filter out unknown intents
        known_df = df[df["intent"] != "unknown"]

        # Intent filter selectbox
        intent_options = ["All"] + sorted(
            [str(i) for i in known_df["intent"].dropna().unique()]
        )
        selected_intent = st.selectbox("🔍 Filter by Intent", intent_options)

        # Filter by selected intent
        if selected_intent != "All":
            filtered_df = known_df[known_df["intent"] == selected_intent]
        else:
            filtered_df = df

        # Group and count by time_of_day
        time_counts = filtered_df["time_of_day"].value_counts().reindex(
            ["Morning ☀️", "Afternoon 🌤️", "Evening 🌇", "Night 🌙"], fill_value=0
        ).reset_index()
        time_counts.columns = ["Time of Day", "Messages"]

        # Chart
        st.markdown("### 📊 Chat Logs by Time of Day")
        chart = alt.Chart(time_counts).mark_bar(
            cornerRadiusTopLeft=8,
            cornerRadiusTopRight=8
        ).encode(
            x=alt.X("Time of Day", sort=["Morning ☀️", "Afternoon 🌤️", "Evening 🌇", "Night 🌙"]),
            y="Messages",
            color=alt.Color("Time of Day", legend=None),
            tooltip=["Time of Day", "Messages"]
        ).properties(
            width=600,
            height=400
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )

        st.altair_chart(chart, use_container_width=True)

        
        st.markdown(" ")
        st.markdown(" ")
        
        with st.expander("💬 Open Chat Assistant"):
            st.markdown("### 🤖 Ask the Assistant")
            
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

            # Prepare chat context from your processed chat_logs, including intents
            context_messages = "\n".join(
                [
                    f"User (Intent: {row['intent']}): {row['user_message']}\nAssistant: {row['assistant_response']}"
                    for _, row in df.iterrows()
                ]
            )

            # Initialize Gemini model
            model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")

            # Chat history state
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Show previous messages
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Get user input
            prompt = st.chat_input("Ask about trends, messages, or topics...")

            if prompt:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Formulate the full prompt
                full_prompt = (
                    "You are an assistant analyzing chat logs between Grab drivers and a voice assistant.\n\n"
                    f"Here are some example logs:\n{context_messages[:6000]}\n\n"
                    f"Now answer this user query: {prompt}"
                )

                # Get the assistant's response
                response = model.generate_content(full_prompt)
                reply = response.text

                # Store the response in chat history
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                with st.chat_message("assistant"):
                    st.markdown(reply)



