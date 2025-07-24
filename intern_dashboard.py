# intern_dashboard.py
# A clean, layered Streamlit dashboard to analyze sunglasses data across brands and retailers
# Author: Khushboo Agrawal (Summer 2025)
# Purpose: Built for a business audience to explore SKU counts, silhouette styles, average pricing, and lens technologies
# Data Source: Cleaned Excel file with one sheet per retailer

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -------------------------------------------------------------------------------------
# 1. Load and clean data from Excel
# -------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Loads and concatenates all retailer sheets from the Excel workbook.
    Ensures clean column naming and type standardization across all sheets.
    Adds a 'Retailer' column inferred from the sheet name.
    Normalizes lens technology.
    """
    def normalize_lens_tech(val):
        # Normalize lens technology values to a standard set
        original_val = str(val)
        val = str(val).lower()
        if 'polarized & uva/uvb block' in val:
            result = 'Polarized + UVA/UVB Block'
        elif 'polarized' in val:
            result = 'Polarized'
        elif 'uva/uvb block' in val:
            result = 'UVA/UVB Block'
        elif 'uv' in val or 'block' in val:
            result = 'UVA/UVB Block'
        else:
            result = 'None'
        return result

    # Load all sheets from the Excel file
    xls = pd.ExcelFile("internDashbaordData.xlsx")
    print(f"\n=== DATA LOADING DEBUG ===")
    print(f"Found {len(xls.sheet_names)} sheets: {xls.sheet_names}")
    df_list = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        # Add retailer name as a column (sheet name)
        if isinstance(df, pd.DataFrame):
            df['Retailer'] = sheet.strip().title()
            print(f" Sheet '{sheet}': {len(df)} rows, columns: {list(df.columns)}")
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset: {len(df_all)} total rows")
    print("=== END DATA LOADING DEBUG ===\n")

    # Clean up column names and types
    df_all.columns = df_all.columns.str.strip()
    df_all['Brand'] = df_all['Brand'].astype(str).str.strip().str.title()
    df_all['Silhouette'] = df_all['Silhouette'].astype(str).str.strip().str.title()
    df_all['Retailer'] = df_all['Retailer'].astype(str).str.strip().str.title()
    df_all['Lens Tech'] = df_all['Lens Technology i.e. Polarized, Drivers, UVA/UVB, etc.'].apply(normalize_lens_tech)
    df_all['Price'] = pd.to_numeric(df_all['AVERAGE SILHOUETTE PRICE'], errors='coerce')

    # Drop rows missing essential info
    return df_all.dropna(subset=['Brand', 'Silhouette'])

# -------------------------------------------------------------------------------------
# 2. Summary helper for verbal chart descriptions
# -------------------------------------------------------------------------------------
def describe_top_silhouettes(df):
    """
    Returns a markdown-formatted sentence summarizing the top 3 silhouettes
    using 'Silhouette count' instead of row frequency.
    """
    grouped = df.groupby("Silhouette", as_index=False)["Silhouette count"].sum()
    grouped = grouped.sort_values(by="Silhouette count", ascending=False)
    total = grouped["Silhouette count"].sum()
    top = grouped.head(3)
    if total == 0 or top.empty:
        return "No silhouette data available."
    parts = [f"**{row['Silhouette']}** ({row['Silhouette count'] / total:.1%})"
             for _, row in top.iterrows()]
    return "Top silhouettes: " + ", ".join(parts) + "."

def describe_top(series, item_type):
    """
    Generic function to describe top items in a series
    """
    counts = series.value_counts()
    total = len(series)
    if total == 0:
        return f"No {item_type} data available."
    top_3 = counts.head(3)
    parts = [f"**{item}** ({count/total:.1%})" for item, count in top_3.items()]
    return f"Top {item_type}s: " + ", ".join(parts) + "."

# -------------------------------------------------------------------------------------
# 3. Helper function for price bucket creation
# -------------------------------------------------------------------------------------
def create_price_bucket(price):
    if pd.isna(price):
        return 'Unknown'
    elif price <= 25:
        return '$0–$25'  # Value segment (includes $25)
    elif price <= 50:
        return '$25–$50'  # Mid-market segment (>$25 up to $50)
    else:
        return '$50+'  # Premium segment (>$50)

# -------------------------------------------------------------------------------------
# 4. Market-Level Overview Page
# -------------------------------------------------------------------------------------
def show_market_page(df):
    """
    Displays the Market Analysis dashboard with silhouette breakdown, SKU counts, pricing, and lens technology distribution.
    """
    st.title("🌍 Market-Wide Overview")
    # Add descriptive paragraph for context
    st.markdown("""
    This section analyzes the overall sunglasses market assortment and retailer partnerships across price points and styles. Explore SKU distribution, silhouette mix, and brand-retailer relationships to uncover market strengths and optimization opportunities.
    """)
    # --- Hero Metrics for Market Analysis ---
    # 1. Total SKUs: sum of Silhouette count across all sheets
    total_skus = df['Silhouette count'].sum()
    # 2. Total Retailers: number of sheets in the Excel file
    try:
        xls = pd.ExcelFile("internDashbaordData.xlsx")
        total_retailers = len(xls.sheet_names)
    except Exception:
        total_retailers = df['Retailer'].nunique()  # fallback
    # 3. Total Brands: unique brands across all sheets
    try:
        unique_brands = set()
        for sheet in xls.sheet_names:
            sheet_df = xls.parse(sheet)
            # Ensure sheet_df is a DataFrame and 'Brand' is a column
            if isinstance(sheet_df, pd.DataFrame) and 'Brand' in sheet_df.columns:
                unique_brands.update(sheet_df['Brand'].astype(str).str.strip().str.title().unique())
        total_brands = len(unique_brands)
    except Exception:
        total_brands = df['Brand'].nunique()  # fallback
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total SKUs", int(total_skus))
    col2.metric("Total Brands", int(total_brands))
    col3.metric("Total Retailers", int(total_retailers))
    st.divider()

    # --- Silhouette Breakdown Pie Chart ---
    st.subheader("Silhouette Breakdown Across All Retailers")
    sil_counts = df.groupby('Silhouette')['Silhouette count'].sum().reset_index()
    sil_counts = sil_counts[sil_counts['Silhouette'].str.lower() != 'nan']
    st.markdown(describe_top_silhouettes(df))
    sil_counts = df.groupby('Silhouette')['Silhouette count'].sum().reset_index()
    fig1 = px.pie(
        sil_counts,
        names='Silhouette',
        values='Silhouette count',
        hole=0.4
    )
    fig1.update_traces(
        textinfo='percent+label',
        textposition='inside',
        insidetextorientation='auto'
    )
    st.plotly_chart(fig1, use_container_width=True)
    # Insight for silhouette breakdown
    if not sil_counts.empty and sil_counts['Silhouette count'].sum() > 0:
        sil_counts_sorted = sil_counts.sort_values('Silhouette count', ascending=False)
        top_silhouette = sil_counts_sorted.iloc[0]
        total_count = sil_counts_sorted['Silhouette count'].sum()
        st.markdown(f"**💡 Insight**: {top_silhouette['Silhouette']} dominates the market with {top_silhouette['Silhouette count']:.0f} units ({top_silhouette['Silhouette count']/total_count:.1%} of total), indicating strong consumer preference for this style.")
    else:
        st.markdown("**💡 Insight**: No silhouette data available for the market.")

    # --- SKU Count by Retailer ---
    st.subheader("SKU Count by Retailer")
    brand_retailer = df[['Retailer', 'Brand', 'Total Brand SKU Count Online']].drop_duplicates()
    sku_counts = brand_retailer.groupby('Retailer')['Total Brand SKU Count Online'].sum().reset_index()
    fig2 = px.bar(sku_counts, x='Retailer', y='Total Brand SKU Count Online', color='Retailer')
    st.plotly_chart(fig2, use_container_width=True)
    # Insight for SKU count by retailer
    if not sku_counts.empty and sku_counts['Total Brand SKU Count Online'].max() > 0:
        top_retailer = sku_counts.loc[sku_counts['Total Brand SKU Count Online'].idxmax()]
        st.markdown(f"**💡 Insight**: {top_retailer['Retailer']} leads with {top_retailer['Total Brand SKU Count Online']:.0f} SKUs, suggesting they offer the widest variety of sunglasses in the market.")
    else:
        st.markdown("**💡 Insight**: No SKU count data available for retailers.")

    # --- Average Price by Brand ---
    st.subheader("Average Price by Brand")
    avg_price = df.groupby('Brand')['Price'].mean().sort_values(ascending=False).reset_index()
    fig3 = px.bar(avg_price, x='Brand', y='Price', color='Brand', height=500)
    st.plotly_chart(fig3, use_container_width=True)
    # Insight for average price by brand
    if not avg_price.empty and avg_price['Price'].notna().any():
        premium_brand = avg_price[avg_price['Price'].notna()].iloc[0] if avg_price['Price'].notna().any() else None
        value_brand = avg_price[avg_price['Price'].notna()].iloc[-1] if avg_price['Price'].notna().any() else None
        if premium_brand is not None and value_brand is not None:
            st.markdown(f"**💡 Insight**: {premium_brand['Brand']} commands the highest average price at ${premium_brand['Price']:.1f}, while {value_brand['Brand']} offers the best value at ${value_brand['Price']:.1f} average price.")
        else:
            st.markdown("**💡 Insight**: No valid price data available for brands.")
    else:
        st.markdown("**💡 Insight**: No price data available for brands.")

    # --- Lens Technology Distribution ---
    st.subheader("Lens Technology Distribution")
    # Aggregate lens technology data by summing silhouette counts
    tech_agg = df.groupby('Lens Tech')['Silhouette count'].sum().reset_index()
    tech_agg = tech_agg.sort_values('Silhouette count', ascending=False)
    fig4 = px.bar(tech_agg, x='Lens Tech', y='Silhouette count', height=500)
    fig4.update_layout(yaxis_title='Total Silhouette Count')
    st.plotly_chart(fig4, use_container_width=True)
    # Insight for lens technology distribution
    if not tech_agg.empty and tech_agg['Silhouette count'].sum() > 0:
        top_tech = tech_agg.iloc[0]
        total_tech = tech_agg['Silhouette count'].sum()
        st.markdown(f"**💡 Insight**: {top_tech['Lens Tech']} is the most popular lens technology with {top_tech['Silhouette count']:.0f} units ({top_tech['Silhouette count']/total_tech:.1%} of total), indicating strong consumer demand for this protection level.")
    else:
        st.markdown("**💡 Insight**: No lens technology data available.")

    # --- Best Value Brands and Retailers: Bubble Chart ---
    st.subheader("🟠 Best Value Brands and Retailers")
    # For each brand, for each retailer, only count the SKU once (first occurrence)
    unique_sku = df.drop_duplicates(subset=['Brand', 'Retailer'])[['Brand', 'Retailer', 'Total Brand SKU Count Online']]
    sku_sum = unique_sku.groupby('Brand')['Total Brand SKU Count Online'].sum().reset_index()
    silhouette_sum = df.groupby('Brand')['Silhouette count'].sum().reset_index()
    avg_price = df.groupby('Brand')['Price'].mean().reset_index()
    # Merge all together
    brand_value = avg_price.merge(sku_sum, on='Brand').merge(silhouette_sum, on='Brand')
    brand_value.columns = ['Brand', 'Avg_Price', 'Total_SKUs', 'Total_Silhouette_Count']
    # Filter out brands with less than 6 total silhouette count for cleaner visualization
    brand_value_filtered = brand_value[brand_value['Total_Silhouette_Count'] >= 6]
    # Filter out brands with NaN average price for insight logic
    brand_value_valid = brand_value_filtered[brand_value_filtered['Avg_Price'].notna()]
    # Create bubble chart for brands
    fig5 = px.scatter(
        brand_value_filtered,
        x='Avg_Price',
        y='Total_SKUs',
        size='Total_Silhouette_Count',
        color='Brand',
        hover_data=['Brand', 'Avg_Price', 'Total_SKUs', 'Total_Silhouette_Count'],
        title='Brand Value Analysis: Price vs SKU Count (Min. 6 Silhouettes)'
    )
    fig5.update_layout(
        xaxis_title='Average Price ($)',
        yaxis_title='Total SKU Count',
        height=500
    )
    st.plotly_chart(fig5, use_container_width=True)
    # Insight for brand value analysis
    if not brand_value_valid.empty:
        best_value_brand = brand_value_valid.loc[brand_value_valid['Total_SKUs'].idxmax()]
        premium_brand = brand_value_valid.loc[brand_value_valid['Avg_Price'].idxmax()]
        st.markdown(f"**💡 Insight**: {best_value_brand['Brand']} offers the best variety with {best_value_brand['Total_SKUs']:.0f} SKUs at ${best_value_brand['Avg_Price']:.1f} average price, while {premium_brand['Brand']} positions as premium at ${premium_brand['Avg_Price']:.1f} average price.")
    else:
        st.markdown("**💡 Insight**: No valid brand value data available for this chart.")

    # --- Retailer Price Bucket Distribution: Stacked Bar Chart ---
    st.subheader("🔵 Retailer Price Segment Mix")
    # Create price buckets for each product
    if 'Price_Bucket' not in df.columns:
        df['Price_Bucket'] = df['Price'].apply(create_price_bucket)
    # Weighted method: sum Silhouette count per bucket
    price_bucket_counts = df.groupby(['Retailer', 'Price_Bucket'])['Silhouette count'].sum().reset_index(name='Count')
    total_per_retailer = price_bucket_counts.groupby('Retailer')['Count'].sum().reset_index(name='Total')
    price_bucket_counts = price_bucket_counts.merge(total_per_retailer, on='Retailer')
    price_bucket_counts['Percent'] = 100 * price_bucket_counts['Count'] / price_bucket_counts['Total']
    # Pivot for stacked bar chart (percentages)
    pivot = price_bucket_counts.pivot(index='Retailer', columns='Price_Bucket', values='Percent').fillna(0)
    bucket_order = ['$0–$25', '$25–$50', '$50+']
    pivot = pivot.reindex(columns=bucket_order, fill_value=0)
    fig = go.Figure()
    for bucket in bucket_order:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[bucket],
            name=bucket,
            text=[f"{v:.1f}%" for v in pivot[bucket]],
            textposition='inside',
            hovertemplate='Retailer: %{x}<br>Percent: %{y:.1f}%<extra></extra>'
        ))
    fig.update_layout(
        barmode='stack',
        xaxis_title='Retailer',
        yaxis_title='Percent of Products (Weighted by Silhouette Count)',
        yaxis=dict(range=[0, 100]),
        title='Product Price Segment Mix by Retailer (Weighted by Silhouette Count)',
        height=600,
        legend_title='Price Bucket',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(b=200)
    )
    st.plotly_chart(fig, use_container_width=True)
    # Insight for price segment mix
    if not price_bucket_counts.empty and price_bucket_counts['Count'].sum() > 0:
        st.markdown("**💡 Insight**: This chart shows the price segment mix for each retailer, weighted by silhouette count. Hover to see the exact percentage in each segment.")
    else:
        st.markdown("**💡 Insight**: No price segment data available for retailers.")

    # --- Drill Down Navigation ---
    st.divider()
    st.subheader("Drill Down")
    col1, col2 = st.columns(2)
    with col1:
        selected_retailer = st.selectbox("Select a Retailer", sorted(df['Retailer'].unique()))
        if st.button("Explore Retailer"):
            st.session_state.view = 'retailer'
            st.session_state.choice = selected_retailer
    with col2:
        selected_brand = st.selectbox("Select a Brand", sorted(df['Brand'].unique()))
        if st.button("Explore Brand"):
            st.session_state.view = 'brand'
            st.session_state.choice = selected_brand

# -------------------------------------------------------------------------------------
# 5. Retailer-Level Page
# -------------------------------------------------------------------------------------
def show_retailer_page(df, retailer):
    st.title(f"🏪 Retailer: {retailer}")
    df_r = df[df['Retailer'] == retailer]

    st.subheader("Silhouette Distribution")
    # Aggregate silhouette data by summing silhouette counts
    sil_agg = df_r.groupby('Silhouette')['Silhouette count'].sum().reset_index()
    sil_agg = sil_agg.sort_values('Silhouette count', ascending=False)
    # Create description for top silhouettes
    total_count = sil_agg['Silhouette count'].sum()
    top_silhouettes = sil_agg.head(3)
    parts = [f"**{row['Silhouette']}** ({row['Silhouette count']/total_count:.1%})" for _, row in top_silhouettes.iterrows()]
    st.markdown("Top silhouettes: " + ", ".join(parts) + ".")
    fig1 = px.pie(sil_agg, names='Silhouette', values='Silhouette count', hole=0.4)
    st.plotly_chart(fig1, use_container_width=True)
    # Insights for retailer silhouette distribution
    if not sil_agg.empty:
        # The data is already sorted by silhouette count descending
        top_silhouette = sil_agg.iloc[0]
        total_count = sil_agg['Silhouette count'].sum()
        st.markdown(f"**💡 Insight**: {top_silhouette['Silhouette']} is the most popular silhouette at {retailer} with {top_silhouette['Silhouette count']:.0f} units ({top_silhouette['Silhouette count']/total_count:.1%} of their inventory).")

    st.subheader("Average Price per Silhouette")
    price = df_r.groupby('Silhouette')['Price'].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(price, x='Silhouette', y='Price', color='Silhouette')
    st.plotly_chart(fig2, use_container_width=True)
    # Insights for retailer average price per silhouette
    if not price.empty:
        # The data is already sorted by price descending
        premium_silhouette = price.iloc[0]
        value_silhouette = price.iloc[-1]
        st.markdown(f"**💡 Insight**: {premium_silhouette['Silhouette']} commands the highest average price at ${premium_silhouette['Price']:.1f}, while {value_silhouette['Silhouette']} offers the best value at ${value_silhouette['Price']:.1f} at {retailer}.")

    st.subheader("Brand Distribution")
    # Aggregate brand data by summing silhouette counts
    brand_agg = df_r.groupby('Brand')['Silhouette count'].sum().reset_index()
    brand_agg = brand_agg.sort_values('Silhouette count', ascending=False)
    # Create description for top brands
    total_count = brand_agg['Silhouette count'].sum()
    top_brands = brand_agg.head(3)
    parts = [f"**{row['Brand']}** ({row['Silhouette count']/total_count:.1%})" for _, row in top_brands.iterrows()]
    st.markdown("Top brands: " + ", ".join(parts) + ".")
    fig3 = px.pie(brand_agg, names='Brand', values='Silhouette count', hole=0.4)
    st.plotly_chart(fig3, use_container_width=True)
    # Insights for retailer brand distribution
    if not brand_agg.empty:
        # The data is already sorted by silhouette count descending
        top_brand = brand_agg.iloc[0]
        total_count = brand_agg['Silhouette count'].sum()
        st.markdown(f"**💡 Insight**: {top_brand['Brand']} is the dominant brand at {retailer} with {top_brand['Silhouette count']:.0f} units ({top_brand['Silhouette count']/total_count:.1%} of their inventory), indicating strong brand partnership.")

    st.subheader("Lens Technology Usage")
    # Aggregate lens technology data by summing silhouette counts
    tech_agg = df_r.groupby('Lens Tech')['Silhouette count'].sum().reset_index()
    tech_agg = tech_agg.sort_values('Silhouette count', ascending=False)
    fig4 = px.bar(tech_agg, x='Lens Tech', y='Silhouette count')
    fig4.update_layout(yaxis_title='Total Silhouette Count')
    st.plotly_chart(fig4, use_container_width=True)
    # Insights for retailer lens technology usage
    if not tech_agg.empty:
        # The data is already sorted by silhouette count descending
        top_tech = tech_agg.iloc[0]
        total_tech = tech_agg['Silhouette count'].sum()
        st.markdown(f"**💡 Insight**: {top_tech['Lens Tech']} is the most popular lens technology at {retailer} with {top_tech['Silhouette count']:.0f} units ({top_tech['Silhouette count']/total_tech:.1%} of their inventory), reflecting consumer preferences for this protection level.")

# -------------------------------------------------------------------------------------
# 6. Brand-Level Page
# -------------------------------------------------------------------------------------
def show_brand_page(df, brand):
    st.title(f"🕶️ Brand: {brand}")
    df_b = df[df['Brand'] == brand]

    st.subheader("Retailer Distribution for Brand")
    # Aggregate retailer data by summing silhouette counts
    retailer_agg = df_b.groupby('Retailer')['Silhouette count'].sum().reset_index()
    retailer_agg = retailer_agg.sort_values('Silhouette count', ascending=False)
    # Create description for top retailers
    total_count = retailer_agg['Silhouette count'].sum()
    top_retailers = retailer_agg.head(3)
    parts = [f"**{row['Retailer']}** ({row['Silhouette count']/total_count:.1%})" for _, row in top_retailers.iterrows()]
    st.markdown("Top retailers: " + ", ".join(parts) + ".")
    fig1 = px.pie(retailer_agg, names='Retailer', values='Silhouette count', hole=0.4)
    st.plotly_chart(fig1, use_container_width=True)
    # Insights for brand retailer distribution
    if not retailer_agg.empty:
        # The data is already sorted by silhouette count descending
        top_retailer = retailer_agg.iloc[0]
        total_count = retailer_agg['Silhouette count'].sum()
        st.markdown(f"**💡 Insight**: {top_retailer['Retailer']} is the primary distribution partner for {brand} with {top_retailer['Silhouette count']:.0f} units ({top_retailer['Silhouette count']/total_count:.1%} of their total inventory), indicating strong retail relationship.")

    st.subheader("SKU Count by Silhouette")
    # Aggregate silhouette data by summing silhouette counts
    sku_agg = df_b.groupby('Silhouette')['Silhouette count'].sum().reset_index()
    sku_agg = sku_agg.sort_values('Silhouette count', ascending=False)
    fig2 = px.bar(sku_agg, x='Silhouette', y='Silhouette count', color='Silhouette')
    fig2.update_layout(yaxis_title='Total Silhouette Count')
    st.plotly_chart(fig2, use_container_width=True)
    # Insights for brand SKU count by silhouette
    if not sku_agg.empty:
        # The data is already sorted by silhouette count descending
        top_silhouette = sku_agg.iloc[0]
        total_count = sku_agg['Silhouette count'].sum()
        st.markdown(f"**💡 Insight**: {top_silhouette['Silhouette']} is {brand}'s most popular silhouette with {top_silhouette['Silhouette count']:.0f} units ({top_silhouette['Silhouette count']/total_count:.1%} of their inventory), suggesting this is their signature style.")

    st.subheader("Average Price by Silhouette")
    price = df_b.groupby('Silhouette')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
    fig3 = px.bar(price, x='Silhouette', y='Price', color='Silhouette')
    st.plotly_chart(fig3, use_container_width=True)
    # Insights for brand average price by silhouette
    if not price.empty:
        # The data is already sorted by price descending
        premium_silhouette = price.iloc[0]
        value_silhouette = price.iloc[-1]
        st.markdown(f"**💡 Insight**: {premium_silhouette['Silhouette']} is {brand}'s premium offering at ${premium_silhouette['Price']:.1f} average price, while {value_silhouette['Silhouette']} serves as their value option at ${value_silhouette['Price']:.1f}.")

# -------------------------------------------------------------------------------------
# Internal Analysis Section
# -------------------------------------------------------------------------------------
def load_internal_data():
    """
    Loads internal assortment and pricing data from InternalData.xlsx.
    Returns two DataFrames: shape_df (Shape sheet), pricing_df (Pricing sheet)
    """
    xls = pd.ExcelFile("InternalData.xlsx")
    shape_df = xls.parse("Shape")
    pricing_df = xls.parse("Pricing")
    return shape_df, pricing_df

def show_internal_analysis():
    st.title("🕶️ Internal Sunglasses Assortment Overview")
    st.markdown("""
    This section analyzes our internal sunglasses assortment and retailer partnerships across price points and styles. Explore SKU distribution, silhouette mix, and brand-retailer relationships to uncover assortment strengths and optimization opportunities.
    """)
    # Load data
    shape_df, pricing_df = load_internal_data()
    # --- Hero metrics ---
    # Ensure 'Account' is the retailer column
    if not isinstance(pricing_df, pd.DataFrame) or 'Account' not in pricing_df.columns:
        st.error("'Account' column not found in Pricing sheet.")
        return
    # SKU count: count all non-empty cells in shape_df
    if not isinstance(shape_df, pd.DataFrame):
        st.error("Shape sheet not loaded as DataFrame.")
        return
    sku_count = shape_df.notnull().sum().sum()
    # Brand count: number of columns in shape_df (excluding index if present)
    brand_count = len(shape_df.columns)
    # Retailer count: number of unique retailers in pricing_df
    retailer_count = pricing_df['Account'].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total SKUs", int(sku_count))
    col2.metric("Total Brands", int(brand_count))
    col3.metric("Total Retailers", int(retailer_count))
    st.divider()
    # --- 1. Pie Chart: SKU Count by Brand ---
    st.subheader("SKU Count by Brand")
    brand_sku_counts = shape_df.notnull().sum(axis=0)
    fig1 = px.pie(
        names=brand_sku_counts.index,
        values=brand_sku_counts.values,
        title="SKU Distribution by Brand"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(f"**Insight:** {brand_sku_counts.idxmax()} has the largest share of SKUs in the internal assortment.")
    # --- 2. Pie Chart: Silhouette Type Distribution ---
    st.subheader("Silhouette Type Distribution")
    silhouettes = pd.Series(shape_df.values.flatten()).dropna().astype(str).str.strip()
    silhouette_counts = silhouettes.value_counts()
    fig2 = px.pie(
        names=silhouette_counts.index,
        values=silhouette_counts.values,
        title="Silhouette Type Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"**Insight:** {silhouette_counts.idxmax()} is the most common silhouette type in the internal assortment.")
    # --- 3. Stacked Bar: Brand Presence Across Retailers ---
    st.subheader("Brand Presence Across Retailers")
    # Extract retailers and brands as described
    retailers = pricing_df['Account'].dropna().tolist()
    brands = [col for col in pricing_df.columns if col != 'Account']
    # Build long-form DataFrame: each row = (Brand, Retailer) if cell is not empty
    brand_retailer_map = []
    for brand in brands:
        for idx, price in pricing_df[brand].items():
            # Ensure price is a scalar, not a Series
            if isinstance(price, (int, float, str)) and not pd.isna(price):
                retailer = pricing_df.at[idx, 'Account']
                brand_retailer_map.append({'Brand': brand, 'Retailer': retailer})
    brand_retailer_df = pd.DataFrame(brand_retailer_map)
    # Pivot for stacked bar chart: brands on x, retailers as stacked segments
    if not brand_retailer_df.empty:
        pivot = pd.crosstab(index=brand_retailer_df['Brand'], columns=brand_retailer_df['Retailer'])
        # Plot as stacked bar chart
        fig3 = go.Figure()
        for retailer in pivot.columns:
            fig3.add_trace(go.Bar(
                x=pivot.index,
                y=pivot[retailer],
                name=retailer,
                hovertemplate=f"{retailer}<extra></extra>"
            ))
        fig3.update_layout(
            barmode='stack',
            xaxis_title='Brand',
            yaxis_title='Number of Retailers',
            title='Brand Presence Across Retailers (Each Segment = 1 Retailer)',
            height=600,
            legend_title='Retailer',
            xaxis_tickangle=-45,
            margin=dict(b=200)
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**Insight:** For each brand, the bar shows which retailers carry it. Hover to see the retailer.")
    else:
        st.info("No brand-retailer relationships found in Pricing sheet.")

    # --- Brand Drill-Down ---
    st.divider()
    st.subheader("Brand Drill-Down")
    brand_list = [col for col in shape_df.columns if col not in ["", None]]
    selected_brand = st.selectbox("Select a Brand", brand_list, key="brand_drilldown")

    # Pie Chart: Silhouette Breakdown
    silhouettes = shape_df[selected_brand].dropna().astype(str).str.strip()
    silhouette_counts = silhouettes.value_counts()
    if silhouette_counts.size > 0:
        fig_pie = px.pie(
            names=silhouette_counts.index,
            values=silhouette_counts.values,
            title=f"Silhouette Breakdown for {selected_brand}"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown(f"**Insight:** This shows what silhouette types make up {selected_brand}'s assortment (e.g., 50% Square, 30% Aviator, etc.). It gives insight into their style positioning.")
    else:
        st.info("No silhouette data for this brand.")

    # Bar Chart: Price by Retailer
    if (isinstance(selected_brand, str) 
        and selected_brand is not None 
        and selected_brand != '' 
        and selected_brand in pricing_df.columns):
        price_data = pricing_df[["Account", selected_brand]]
        # Only call dropna if selected_brand is a valid, non-None, non-empty column
        if selected_brand is not None and selected_brand != '':
            price_data = price_data.dropna(subset=[selected_brand])
        price_data = price_data.rename(columns={selected_brand: "MSRP"})
        if not price_data.empty:
            fig_bar = px.bar(
                price_data,
                x="MSRP",
                y="Account",
                orientation="h",
                title=f"MSRP by Retailer for {selected_brand}",
                labels={"Account": "Retailer", "MSRP": "MSRP"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown(f"**Insight:** This shows which retailers carry {selected_brand} and at what MSRP.")
        else:
            st.info("No pricing data for this brand.")
    else:
        st.info("No pricing data for this brand.")

# -------------------------------------------------------------------------------------
# 7. MAIN STREAMLIT CONTROLLER (updated)
# -------------------------------------------------------------------------------------
def main():
    st.set_page_config("Intern Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ("Market Analysis", "Internal Analysis")
    )
    if 'view' not in st.session_state:
        st.session_state.view = 'market'
        st.session_state.choice = None
    if page == "Market Analysis":
        df = load_data()
        if st.session_state.view == 'market':
            show_market_page(df)
        elif st.session_state.view == 'retailer':
            show_retailer_page(df, st.session_state.choice)
        elif st.session_state.view == 'brand':
            show_brand_page(df, st.session_state.choice)
        st.sidebar.button("🏠 Home", on_click=lambda: st.session_state.update({'view': 'market'}))
    elif page == "Internal Analysis":
        show_internal_analysis()

if __name__ == "__main__":
    main()