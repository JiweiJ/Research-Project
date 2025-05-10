import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import re
from datetime import datetime, timedelta

FILENAME_PATTERN = re.compile(r'post(\d{8})\d{4}')
DATA_DIR = r'C:\\Users\\ENJD\\Desktop\\Bluesky'
PREDICTION_ROOT = r'C:\\Users\\ENJD'

st.set_page_config(page_title="Bluesky Post Analysis", layout="wide")
st.title("Bluesky Post Data Visualization")

@st.cache_data
def load_all_data(folder_path):
    dfs = []
    for file in os.listdir(folder_path):
        if file.startswith('post') and file.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(folder_path, file))
            ts = file.replace('post', '').replace('.xlsx', '')
            df['Capture Time'] = datetime.strptime(ts, "%Y%m%d%H%M")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


df = load_all_data(DATA_DIR)
if df.empty:
    st.error("No data loaded!")
    st.stop()

if 'Capture Time' in df.columns:
    df['Capture Date'] = df['Capture Time'].dt.date
else:
    df['Post Time'] = pd.to_datetime(df['Post Time'], utc=True, errors='coerce')
    df.dropna(subset=['Post Time'], inplace=True)
    df['Post Time'] = df['Post Time'].dt.tz_localize(None)
    df['Capture Date'] = df['Post Time'].dt.date

st.sidebar.header("Time")
min_date = df['Capture Date'].min()
max_date = df['Capture Date'].max()
start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

df_filtered = df[(df['Capture Date'] >= start_date) & (df['Capture Date'] <= end_date)]

unique_total_posts = df_filtered.drop_duplicates(subset=['Post ID'])
st.markdown(f"**Total Posts:** {len(unique_total_posts)}")


# post by query
if 'Query' in df_filtered.columns:
    query_unique_posts = df_filtered.drop_duplicates(subset=['Post ID']).groupby('Query')['Post ID'].count().sort_values(ascending=False)
    st.subheader('Unique Post Count per Query')
    st.dataframe(query_unique_posts.rename("Unique Post Count"))

    fig_query_pie = px.pie(values=query_unique_posts.values, names=query_unique_posts.index, title='Unique Post Share by Query')
    st.plotly_chart(fig_query_pie, use_container_width=True, key='unique_query_pie')

# engagement 
if {'Post Like Count', 'Comment Count', 'Repost Count', 'Query'}.issubset(df_filtered.columns):
    df_filtered['Total Engagement'] = (
        df_filtered['Post Like Count'] +
        df_filtered['Comment Count'] +
        df_filtered['Repost Count']
    )
    query_engagement = df_filtered.groupby('Query')['Total Engagement'].sum().sort_values(ascending=False)
    st.subheader('Total Engagement per Query')
    fig_engagement_query = px.bar(query_engagement.reset_index(), x='Query', y='Total Engagement',
                                  labels={'Query':'Query', 'Total Engagement':'Engagement'},
                                  title='Total Engagement by Query')
    st.plotly_chart(fig_engagement_query, use_container_width=True, key='query_total_engagement')

    # daily engagement by query
    daily_query_stats = df_filtered.groupby(['Capture Date', 'Query']).agg(
        Post_Count=('Post ID', 'count'),
        Daily_Engagement=('Total Engagement', 'sum')
    ).reset_index()
    filtered_stats = daily_query_stats[daily_query_stats['Post_Count'] >= 20]

    st.subheader("Daily Engagement for Queries with ≥20 Posts")
    pivot_df = filtered_stats.pivot(index='Capture Date', columns='Query', values='Daily_Engagement')
    fig_daily_eng = px.line(
        pivot_df,
        title="Daily Engagement by Keyword (≥20 Posts)",
        labels={'value': 'Engagement', 'Capture Date': 'Date'}
    )
    fig_daily_eng.update_yaxes(tickvals=[0, 1000, 2000, 5000, 10000, 20000, 50000])
    st.plotly_chart(fig_daily_eng, use_container_width=True, key='daily_query_engagement')

# future
prediction_files = [f for f in os.listdir(PREDICTION_ROOT) if f.startswith('KEYWORD_PREDICT_') and f.endswith('.xlsx')]
if prediction_files:
    latest_file = sorted(prediction_files)[-1]
    prediction_path = os.path.join(PREDICTION_ROOT, latest_file)
    pred_df = pd.read_excel(prediction_path)

    if {'Query', 'Predicted Engagement', 'Predict Date'}.issubset(pred_df.columns):
        st.subheader('Predicted Keyword Engagement (Next Day)')
        st.dataframe(pred_df.sort_values('Predicted Engagement', ascending=False))

        fig_pred = px.bar(pred_df.sort_values('Predicted Engagement', ascending=False),
                          x='Query', y='Predicted Engagement',
                          title='Predicted Engagement by Keyword for Next Day')
        st.plotly_chart(fig_pred, use_container_width=True, key='keyword_prediction')
else:
    st.sidebar.warning('No keyword prediction files found in root directory.')

# posttime hot map
if 'Post Time' in df.columns:
    df_filtered['Post Time'] = pd.to_datetime(df_filtered['Post Time'], utc=True, errors='coerce')
    df_filtered.dropna(subset=['Post Time'], inplace=True)
    df_filtered['Post Time'] = df_filtered['Post Time'].dt.tz_localize(None)
    df_filtered['Post Hour'] = df_filtered['Post Time'].dt.hour
    df_filtered['Post Date'] = df_filtered['Post Time'].dt.date

    df_filtered_heatmap = df_filtered[df_filtered['Post Date'] >= datetime(2025, 4, 10).date()]
    heatmap_data = df_filtered_heatmap.groupby(['Post Date', 'Post Hour']).size().reset_index(name='Count')
    heatmap_pivot = heatmap_data.pivot(index='Post Hour', columns='Post Date', values='Count').fillna(0)
    st.subheader('Post Time Distribution Heatmap')
    fig_heatmap = px.imshow(heatmap_pivot,
                            labels=dict(x="Post Date", y="Post Hour", color="Post Count"),
                            aspect="auto",
                            title="Post Activity by Hour and Day")
    st.plotly_chart(fig_heatmap, use_container_width=True, key='post_heatmap')

# dailt new
st.subheader('Daily New Posts')
dates = sorted(df['Capture Date'].unique())
history_post_ids = set()
new_post_counts = {}
for date in dates:
    today_ids = set(df[df['Capture Date'] == date]['Post ID'])
    new_posts_today = today_ids - history_post_ids
    new_post_counts[date] = len(new_posts_today)
    history_post_ids.update(today_ids)

newdf = pd.DataFrame({
    'Date': list(new_post_counts.keys()),
    'New Posts': list(new_post_counts.values())
})
fig_new = px.line(newdf, x='Date', y='New Posts', title='Daily New Posts')
st.plotly_chart(fig_new, use_container_width=True, key='new_posts')

# Top 10 Posts by Total Engagement
if {'Post Like Count', 'Comment Count', 'Repost Count'}.issubset(df_filtered.columns):
    df_filtered = df_filtered.drop_duplicates(subset=['Post ID'])
    st.subheader('Top 10 Posts by Total Engagement')
    top10 = df_filtered.nlargest(10, 'Total Engagement')[['Post Content', 'Total Engagement']]
    st.dataframe(top10)

# Query
if 'Query' in df_filtered.columns:
    query_counts = df_filtered['Query'].value_counts()
    fig_pie = px.pie(values=query_counts.values, names=query_counts.index, title='Post Share by Query')
    st.plotly_chart(fig_pie, use_container_width=True, key='pie')

    engagement_sums = df_filtered.groupby('Query')['Total Engagement'].sum()
    fig_eng = px.bar(x=engagement_sums.index, y=engagement_sums.values,
                     labels={'x':'Query','y':'Total Engagement'}, title='Engagement by Query')
    st.plotly_chart(fig_eng, use_container_width=True, key='engagement')

# daily detail
daily = df.groupby('Capture Date').agg(
    Likes=('Post Like Count','sum'),
    Comments=('Comment Count','sum'),
    Reposts=('Repost Count','sum')
).sort_index()
st.subheader('Daily Engagement Metrics')
fig_line = px.line(daily.reset_index(), x='Capture Date', y=['Likes','Comments','Reposts'],
                   labels={'value':'Count','Capture Date':'Date'}, title='Daily Likes, Comments, Reposts')
st.plotly_chart(fig_line, use_container_width=True, key='daily_metrics')

# die
dead_counts = {}
dead_rates = {}
for i in range(1, len(dates)):
    prev_ids = set(df[df['Capture Date'] == dates[i-1]]['Post ID'])
    curr_ids = set(df[df['Capture Date'] == dates[i]]['Post ID'])
    dead = prev_ids - curr_ids
    dead_counts[dates[i]] = len(dead)
    dead_rates[dates[i]] = len(dead) / len(curr_ids) if curr_ids else 0

deaddf = pd.DataFrame({'Date': list(dead_counts.keys()), 'Dead Posts': list(dead_counts.values()), 'Dead Rate': list(dead_rates.values())})
st.subheader('Daily Dead Posts & Rate')
col1, col2 = st.columns(2)
with col1:
    fig_dead = px.line(deaddf, x='Date', y='Dead Posts', title='Daily Dead Posts')
    st.plotly_chart(fig_dead, use_container_width=True, key='dead_posts')
with col2:
    fig_rate = px.line(deaddf, x='Date', y='Dead Rate', title='Daily Dead Rate')
    st.plotly_chart(fig_rate, use_container_width=True, key='dead_rate')

# CN vs EN
if 'Query' in df_filtered.columns:
    df_filtered['Lang'] = df_filtered['Query'].apply(
        lambda x: 'Chinese' if any('\u4e00' <= c <= '\u9fff' for c in str(x)) else 'English'
    )
    lang_stats = df_filtered.groupby('Lang').agg(Post_Count=('Post ID','count'), Engagement=('Total Engagement','sum'))
    fig_lang = px.bar(lang_stats.reset_index(), x='Lang', y=['Post_Count','Engagement'], barmode='group', title='Chinese vs English Query Comparison')
    st.plotly_chart(fig_lang, use_container_width=True, key='lang_comparison')

st.sidebar.info('Refresh to load new files and update metrics.')
