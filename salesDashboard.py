# GLOBAL SUPERSTORE – STREAMLIT DASHBOARD
# Run: streamlit run salesDashboard.py
# Make sure Global_Superstore2.csv is in the same folder.

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(
    page_title="Global Superstore Dashboard",
    # page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS
st.markdown("""
<style>
    /* ── top bar ── */
    [data-testid="stHeader"]  {background: #0f969c;}

    /* ── sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #fafafa;
    }

    /* ── sidebar text and labels ── */
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #000000 !important;
    }

    /* ── sidebar widget icons/dropdowns (optional but recommended) ── */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #000000;
    }
    
    /* Makes input box borders visible against black */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        border: 1px solid #333333;
        background-color: #fafafa;
        color: #000000;
    }
    /* ── KPI box styling ── */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(12,112,117,0.35);
        border-left: 5px solid #0e7490;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px;
        font-weight: 700;
        color: #2c2f38;
        letter-spacing: .5px;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px;
        margin-top: 2px;
    }

    /* ── section titles ── */
    .section-title {
        font-size: 17px; font-weight: 700;
        color: #0e7490; margin-top: 8px; margin-bottom: 4px;
    }
    /* ── KPI section title specifically ── */
    .kpi-title {
        font-size: 17px; font-weight: 700;
        color: #294d61; margin-top: 8px; margin-bottom: 4px;
    }

    /* ── divider ── */
    hr.divider {
        border: none;
        border-top: 2px solid #ccfbf1;
        margin: 6px 0 14px 0;
    }
</style>
""", unsafe_allow_html=True)


# DATA LOADING & CLEANING (cached)
@st.cache_data(show_spinner="Loading & cleaning data …")
def load_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, encoding="latin1")

    # Drop rows with nulls in key numeric columns
    df.dropna(subset=["Sales", "Quantity", "Profit", "Shipping Cost"], inplace=True)

    # Parse mixed date formats
    def parse_dates(series):
        formats = ["%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]
        out = pd.Series([pd.NaT] * len(series), index=series.index)
        for fmt in formats:
            mask = out.isna()
            out[mask] = pd.to_datetime(series[mask], format=fmt, errors="coerce")
        still = out.isna()
        out[still] = pd.to_datetime(series[still], dayfirst=True, errors="coerce")
        return out

    df["Order Date"] = parse_dates(df["Order Date"])
    df["Ship Date"]  = parse_dates(df["Ship Date"])
    df.dropna(subset=["Order Date", "Ship Date"], inplace=True)

    # Remap years safely (handles leap day edge cases)
    YEAR_MAP = {2011: 2021, 2012: 2022, 2013: 2023, 2014: 2024}

    def remap_year(dt_series):
        result = []
        for dt in dt_series:
            new_year = YEAR_MAP.get(dt.year, dt.year)
            try:
                result.append(dt.replace(year=new_year))
            except ValueError:
                # Feb 29 leap day fix → Feb 28
                result.append(dt.replace(year=new_year, day=28))
        return pd.to_datetime(result)

    df["Order Date"] = remap_year(df["Order Date"])
    df["Ship Date"]  = remap_year(df["Ship Date"])

    df.sort_values("Order Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Outlier removal using IQR method (same as notebook)
    def remove_outliers_iqr(df, columns):
        for col in columns:
            Q1    = df[col].quantile(0.25)
            Q3    = df[col].quantile(0.75)
            IQR   = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df    = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    df = remove_outliers_iqr(df, ["Sales", "Profit"])

    # Feature engineering (only used features)
    df["Year"]  = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

    return df


FILE_PATH = "Global_Superstore2.csv"   # ← adjust path if needed
df_raw = load_data(FILE_PATH)


# COLOUR PALETTE & STYLE HELPER
TEAL = px.colors.sequential.Teal
n    = len(TEAL)

def style(fig):
    """Apply uniform layout styling to every chart."""
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f9fafb",
        font_family="Arial",
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(
        shapes=[dict(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#6da5c0", width=1.5),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )]
    )
    return fig


# SIDEBAR – FILTERS
with st.sidebar:
    # st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=60)
    st.markdown("### Global Superstore")
    st.markdown("---")
    st.markdown("###  Filters")

    all_years   = sorted(df_raw["Year"].unique())
    sel_years   = st.multiselect("Year",          all_years,   default=all_years)

    all_markets = sorted(df_raw["Market"].dropna().unique())
    sel_markets = st.multiselect("Market",         all_markets, default=all_markets)

    all_cats    = sorted(df_raw["Category"].unique())
    sel_cats    = st.multiselect("Category",       all_cats,    default=all_cats)

    all_subcats = sorted(
        df_raw.loc[df_raw["Category"].isin(sel_cats), "Sub-Category"].unique()
    )
    sel_subcats = st.multiselect("Sub-Category",  all_subcats, default=all_subcats)

    all_segs    = sorted(df_raw["Segment"].unique())
    sel_segs    = st.multiselect("Segment",        all_segs,    default=all_segs)

    min_sale   = float(df_raw["Sales"].min())
    max_sale   = float(df_raw["Sales"].max())
    sale_range = st.slider(
        "Sales Range ($)",
        min_value=min_sale, max_value=max_sale,
        value=(min_sale, max_sale), step=10.0,
    )

    st.markdown("---")
    st.markdown("### Explore")
    explore_option = st.selectbox(
        "Explore View",
        ["None", "Segment Analysis", "Shipping Analysis", "Market Breakdown"],
        index=0,
    )

    st.markdown("---")
    st.caption("Built by Team Hammer")


# APPLY FILTERS
df = df_raw[
    df_raw["Year"].isin(sel_years)           &
    df_raw["Market"].isin(sel_markets)       &
    df_raw["Category"].isin(sel_cats)        &
    df_raw["Sub-Category"].isin(sel_subcats) &
    df_raw["Segment"].isin(sel_segs)         &
    df_raw["Sales"].between(sale_range[0], sale_range[1])
].copy()

if df.empty:
    st.warning("No data matches your filters. Please adjust the sidebar selections.")
    st.stop()


# PRE-COMPUTE AGGREGATES
monthly_sales = df.groupby("Month", as_index=False)["Sales"].sum()
monthly_sales["Month_dt"] = pd.to_datetime(monthly_sales["Month"])
monthly_sales.sort_values("Month_dt", inplace=True)

# Bar chart – sub-category grouped by category
subcat_bar = df.groupby(["Category", "Sub-Category"], as_index=False)["Sales"].sum()
subcat_bar.sort_values(["Category", "Sales"], ascending=[True, False], inplace=True)

sales_cat    = df.groupby("Category", as_index=False).agg(
    Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"))

sales_subcat = df.groupby("Sub-Category", as_index=False).agg(
    Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"))

country_sales = df.groupby("Country", as_index=False)["Sales"].sum()


# HEADER
st.markdown(
    "<h1 style='text-align:center; color:#05161a;'>"
    "Insights on Global Sales and Profit on Various Categories (2021-2024)</h1>",
    unsafe_allow_html=True,
)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# KPI SECTION – aligned to the 7 visualisations in the dashboard
st.markdown("<div class='kpi-title'> Key Performance Indicators</div>",
            unsafe_allow_html=True)

# ── KPI 1: Monthly Sales Trend → Sales Growth (line chart insight)
# Shows whether sales increased or decreased from 2023 to 2024
sales_2024   = df[df["Year"] == 2024]["Sales"].sum()
sales_2023   = df[df["Year"] == 2023]["Sales"].sum()
sales_growth = ((sales_2024 - sales_2023) / sales_2023 * 100) if sales_2023 else 0
# Delta: positive = green arrow up, negative = red arrow down (delta_color="normal")

# ── KPI 2: Bar Chart → Top Performing Sub-Category by Sales
# Reflects the bar chart showing sub-category breakdown
subcat_totals   = df.groupby("Sub-Category")["Sales"].sum()
top_subcat      = subcat_totals.idxmax()
top_subcat_prev = df[df["Year"] == 2023].groupby("Sub-Category")["Sales"].sum()
top_subcat_curr = df[df["Year"] == 2024].groupby("Sub-Category")["Sales"].sum()
top_subcat_val  = subcat_totals.max()
# Compare top sub-cat sales 2024 vs 2023 for direction indicator
top_subcat_growth = 0
if top_subcat in top_subcat_prev.index and top_subcat in top_subcat_curr.index:
    prev_val = top_subcat_prev[top_subcat]
    curr_val = top_subcat_curr[top_subcat]
    top_subcat_growth = ((curr_val - prev_val) / prev_val * 100) if prev_val else 0

# ── KPI 3: Heat Map → Best Performing Category (highest sales in latest year)
# Reflects the heat map showing Category × Year
cat_2024       = df[df["Year"] == 2024].groupby("Category")["Sales"].sum()
cat_2023       = df[df["Year"] == 2023].groupby("Category")["Sales"].sum()
best_cat       = cat_2024.idxmax() if not cat_2024.empty else "N/A"
best_cat_sales = cat_2024.max()    if not cat_2024.empty else 0
best_cat_prev  = cat_2023.get(best_cat, 0)
best_cat_growth = ((best_cat_sales - best_cat_prev) / best_cat_prev * 100) if best_cat_prev else 0

# ── KPI 4: Scatter + Histogram → Net Profit Margin (2024)
# Reflects profitability visible in scatter and histogram charts
sales_2024_all   = df[df["Year"] == 2024]["Sales"].sum()
profit_2024_all  = df[df["Year"] == 2024]["Profit"].sum()
npm_2024         = (profit_2024_all / sales_2024_all * 100) if sales_2024_all else 0
# Compare to 2023 margin for direction
sales_2023_all   = df[df["Year"] == 2023]["Sales"].sum()
profit_2023_all  = df[df["Year"] == 2023]["Profit"].sum()
npm_2023         = (profit_2023_all / sales_2023_all * 100) if sales_2023_all else 0
npm_delta        = npm_2024 - npm_2023   # positive = margin improved

# ── KPI 5: Bubble + Geo Map → Top Region by Sales
# Reflects geographic and bubble chart market distribution
region_sales  = df.groupby("Region")["Sales"].sum()
top_region    = region_sales.idxmax()
top_region_pct = (region_sales.max() / region_sales.sum() * 100)
# Compare region sales 2024 vs 2023 for direction
reg_2024 = df[df["Year"] == 2024].groupby("Region")["Sales"].sum()
reg_2023 = df[df["Year"] == 2023].groupby("Region")["Sales"].sum()
top_reg_growth = 0
if top_region in reg_2024.index and top_region in reg_2023.index:
    top_reg_growth = ((reg_2024[top_region] - reg_2023[top_region])
                      / reg_2023[top_region] * 100) if reg_2023[top_region] else 0

# ── Render all 5 KPIs inside styled boxes
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    # Line chart KPI — sales trend direction shown by delta arrow
    st.metric(
        label="📈 Sales Growth in 2024",
        value=f"{sales_growth:+.1f}%",
        delta=f"${sales_2024:,.0f} in 2024",
        delta_color="normal",   # ↑ green if growth, ↓ red if decline
    )

with k2:
    # Bar chart KPI — top sub-category with direction vs last year
    st.metric(
        label="🏆 Top Selling Product ",
        value=top_subcat,
        delta=f"{top_subcat_growth:+.1f}% vs 2023",
        delta_color="normal",   # ↑ green if sub-cat grew, ↓ red if shrank
    )

with k3:
    # Heat map KPI — best category in 2024 with direction vs 2023
    st.metric(
        label="📦 Best Selling Category",
        value=best_cat,
        delta=f"{best_cat_growth:+.1f}% vs 2023",
        delta_color="normal",   # ↑ green if category grew, ↓ red if declined
    )

with k4:
    # Scatter + histogram KPI — profit margin with direction vs 2023
    st.metric(
        label="💹 Net Profit Margin (2024)",
        value=f"{npm_2024:.1f}%",
        delta=f"{npm_delta:+.1f}pp vs 2023",
        delta_color="normal",   # ↑ green if margin improved, ↓ red if worsened
    )

with k5:
    # Geo + bubble chart KPI — top region with direction vs 2023
    st.metric(
        label="🌍 Top Region (Sales Share)",
        value=f"{top_region_pct:.1f}% — {top_region}",
        delta=f"{top_reg_growth:+.1f}% region growth",
        delta_color="normal",   # ↑ green if region grew, ↓ red if shrank
    )

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ROW 1 – Line Chart | Bar Chart

c1, c2 = st.columns(2)

# ── 1. Line Chart – Monthly Sales Trend
with c1:
    fig_line = px.line(
        monthly_sales,
        x="Month", y="Sales",
        markers=True,
        color_discrete_sequence=[TEAL[int(n * 0.6)]],
        title="Monthly Sales Trend",
        labels={"Sales": "Total Sales ($)", "Month": "Month"},
    )
    fig_line.update_traces(
        hovertemplate="<b>Month:</b> %{x}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>"
    )
    st.plotly_chart(style(fig_line), use_container_width=True)

# ── 2. Bar Chart – Sub-Category Sales grouped by Category
with c2:
    fig_bar = px.bar(
        subcat_bar,
        x="Sales",
        y="Sub-Category",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        orientation="h",
        barmode="group",
        title="Total Sales by Sub-Category",
        labels={
            "Sales"        : "Total Sales ($)",
            "Sub-Category" : "",
            "Category"     : "Category",
        },
        text_auto=".2s",
    )
    fig_bar.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Category: %{fullData.name}<br>"
            "Sales: $%{x:,.0f}<extra></extra>"
        )
    )
    fig_bar.update_layout(
        legend=dict(
            title="Category",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(style(fig_bar), use_container_width=True)

# ROW 2 – Heat Map | Bubble Chart
 
c3, c4 = st.columns(2)
 
# ── 3. Heat Map – Sales by Category × Year
with c3:
    pivot = df.pivot_table(
        index="Category", columns="Year", values="Sales", aggfunc="sum"
    )
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        text_auto=".2s",
        aspect="auto",
        title="Sales – Category × Year",
        labels=dict(x="Year", y="Category", color="Sales ($)"),
    )
    fig_heat.update_traces(
        hovertemplate=(
            "<b>Category:</b> %{y}<br>"
            "<b>Year:</b> %{x}<br>"
            "<b>Sales:</b> $%{z:,.0f}<extra></extra>"
        )
    )
    st.plotly_chart(style(fig_heat), use_container_width=True)
 
# ── 4. Box Plot – Profit by Category (safe colour indexing)
with c4:
    c_b1 = TEAL[int(n * 0.3)]
    c_b2 = TEAL[int(n * 0.6)]
    c_b3 = TEAL[int(n * 0.9)]
 
    fig_box = px.box(
        df,
        x="Category", y="Profit",
        color="Category",
        color_discrete_sequence=[c_b1, c_b2, c_b3],
        points="outliers",
        title="Profit Distribution by Category",
        labels={"Profit": "Profit ($)", "Category": ""},
    )
    fig_box.update_traces(
        hovertemplate=(
            "<b>Category:</b> %{x}<br>"
            "<b>Profit:</b> $%{y:,.0f}<extra></extra>"
        )
    )
    st.plotly_chart(style(fig_box), use_container_width=True)


# ROW 3 – Scatter Plot | Histogram + Density

c5, c6 = st.columns(2)

# ── 5. Scatter Plot – colorful Vivid palette, standalone
with c5:
    fig_scatter = px.scatter(
        df.sample(n=min(2000, len(df)), random_state=42),
        x="Sales",
        y="Profit",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        size="Sales",
        size_max=15,
        opacity=0.7,
        title="Sales vs Profit by Category",
        labels={
            "Sales"    : "Sales ($)",
            "Profit"   : "Profit ($)",
            "Category" : "Category",
        },
        hover_data={"Category": True, "Sub-Category": True},
    )
    fig_scatter.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "<b>Sub-Category:</b> %{customdata[1]}<br>"
            "Sales: $%{x:,.0f}<br>"
            "Profit: $%{y:,.0f}<extra></extra>"
        )
    )
    st.plotly_chart(style(fig_scatter), use_container_width=True)

# ── 6. Histogram + Density – multi-layered, clipped, bin borders
with c6:
    c_h1 = TEAL[int(n * 0.4)]
    c_h2 = TEAL[int(n * 0.7)]
    c_h3 = TEAL[int(n * 0.9)]

    # Clip to 1st–99th percentile to remove extreme outliers
    sales_p01  = df["Sales"].quantile(0.01)
    sales_p99  = df["Sales"].quantile(0.99)
    profit_p01 = df["Profit"].quantile(0.01)
    profit_p99 = df["Profit"].quantile(0.99)

    x_min = min(sales_p01, profit_p01)
    x_max = max(sales_p99, profit_p99)

    sales_clipped  = df["Sales"][
        (df["Sales"]  >= sales_p01)  & (df["Sales"]  <= sales_p99)]
    profit_clipped = df["Profit"][
        (df["Profit"] >= profit_p01) & (df["Profit"] <= profit_p99)]

    fig_hkde = go.Figure()

    # Layer 1 – Histogram with white bin borders
    fig_hkde.add_trace(go.Histogram(
        x=sales_clipped,
        nbinsx=60,
        name="Sales Count",
        marker=dict(
            color=c_h1,
            line=dict(color="white", width=0.8),   # ← bin border
        ),
        opacity=0.6,
        yaxis="y1",
        hovertemplate="Sales: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))

    # Layer 2 – KDE for Sales
    kde_s   = gaussian_kde(sales_clipped, bw_method=0.15)
    x_s     = np.linspace(sales_p01, sales_p99, 500)
    y_s_kde = kde_s(x_s)

    fig_hkde.add_trace(go.Scatter(
        x=x_s, y=y_s_kde,
        name="Sales Density (KDE)",
        line=dict(color=c_h2, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56,166,165,0.12)",
        yaxis="y2",
        hovertemplate="Sales: $%{x:,.0f}<br>Density: %{y:.6f}<extra></extra>",
    ))

    # Layer 3 – KDE for Profit
    kde_p   = gaussian_kde(profit_clipped, bw_method=0.15)
    x_p     = np.linspace(profit_p01, profit_p99, 500)
    y_p_kde = kde_p(x_p)

    fig_hkde.add_trace(go.Scatter(
        x=x_p, y=y_p_kde,
        name="Profit Density (KDE)",
        line=dict(color=c_h3, width=2.5, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(8,104,172,0.10)",
        yaxis="y2",
        hovertemplate="Value: $%{x:,.0f}<br>Density: %{y:.6f}<extra></extra>",
    ))

    fig_hkde.update_layout(
        title=dict(
            text="Histogram + Density – Sales & Profit",
            x=0.5, xanchor="center", font_size=16,
        ),
        xaxis=dict(
            title="Value ($)",
            range=[x_min, x_max],
            tickprefix="$",
            tickformat=",.0f",
        ),
        yaxis=dict(
            title=dict(text="Order Count", font=dict(color=c_h1)),
            tickfont=dict(color=c_h1),
        ),
        yaxis2=dict(
            title=dict(text="Density", font=dict(color=c_h2)),
            tickfont=dict(color=c_h2),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(x=0.65, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
        paper_bgcolor="white",
        plot_bgcolor="#f9fafb",
        font_family="Arial",
        hovermode="x unified",
        margin=dict(t=50, b=30, l=30, r=50),
        barmode="overlay",
    )
    fig_hkde.update_layout(shapes=[dict(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="#6da5c0", width=1.5),
        fillcolor="rgba(0,0,0,0)", layer="above",
    )])
    st.plotly_chart(fig_hkde, use_container_width=True)
    
    
# ROW 4 – Bubble Chart (full width)
 
bubble_df             = sales_subcat.copy()
bubble_df["abs_size"] = np.abs(bubble_df["Total_Sales"])
 
fig_bubble = px.scatter(
    bubble_df,
    x="Total_Sales",
    y="Total_Profit",
    size="abs_size",
    color="Total_Profit",
    color_continuous_scale=TEAL,
    text="Sub-Category",
    size_max=60,
    title="Sub-Category Sales vs Profit",
    labels={
        "Total_Sales"  : "Total Sales ($)",
        "Total_Profit" : "Total Profit ($)",
    },
)
fig_bubble.update_traces(
    textposition="top center",
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Sales: $%{x:,.0f}<br>"
        "Profit: $%{y:,.0f}<extra></extra>"
    ),
)
st.plotly_chart(style(fig_bubble), use_container_width=True)



# ROW 4 – Geographical Map (full width)

fig_map = px.choropleth(
    country_sales,
    locations="Country",
    locationmode="country names",
    color="Sales",
    color_continuous_scale=TEAL,
    projection="natural earth",
    title="Total Sales by Country",
    labels={"Sales": "Total Sales ($)"},
)

# Latitude lines (parallels every 30°)
for lat in range(-90, 91, 30):
    fig_map.add_trace(go.Scattergeo(
        lat=[lat] * 361,
        lon=list(range(-180, 181)),
        mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=0.6),
        showlegend=False,
        hoverinfo="skip",
    ))

# Longitude lines (meridians every 30°)
for lon in range(-180, 181, 30):
    fig_map.add_trace(go.Scattergeo(
        lat=list(range(-90, 91)),
        lon=[lon] * 181,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=0.6),
        showlegend=False,
        hoverinfo="skip",
    ))

# Map geo styling
fig_map.update_geos(
    showcoastlines=True,  coastlinecolor="white",  coastlinewidth=0.8,
    showcountries=True,   countrycolor="white",     countrywidth=0.5,
    showland=True,        landcolor="#f0f0f0",
    showocean=True,       oceancolor="#d6eaf8",
    showlakes=True,       lakecolor="#d6eaf8",
    showrivers=False,
    lataxis=dict(
        showgrid=True, gridcolor="rgba(150,150,150,0.3)",
        gridwidth=0.5, dtick=30,
    ),
    lonaxis=dict(
        showgrid=True, gridcolor="rgba(150,150,150,0.3)",
        gridwidth=0.5, dtick=30,
    ),
)
fig_map.update_traces(
    hovertemplate="<b>%{location}</b><br>Sales: $%{z:,.0f}<extra></extra>",
    selector=dict(type="choropleth"),
)
st.plotly_chart(style(fig_map), use_container_width=True)



# ROW 5 – Funnel Chart | Treemap
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
nc1, nc2 = st.columns(2)

# Funnel – Average Order Value by Segment → Ship Mode
with nc1:
    funnel_df = df.groupby(["Segment", "Ship Mode"], as_index=False).agg(
        Orders=("Order ID", "nunique"),
        Total_Sales=("Sales", "sum"),
    )
    funnel_df["Avg_Order"] = funnel_df["Total_Sales"] / funnel_df["Orders"]
    funnel_df_sorted = funnel_df.groupby("Segment", as_index=False)["Avg_Order"].mean()
    funnel_df_sorted.sort_values("Avg_Order", ascending=False, inplace=True)

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_df_sorted["Segment"],
        x=funnel_df_sorted["Avg_Order"],
        textinfo="value+percent initial",
        texttemplate="%{label}<br>$%{value:,.0f}",
        marker=dict(color=[TEAL[int(n * 0.8)], TEAL[int(n * 0.5)], TEAL[int(n * 0.2)]]),
        hovertemplate="<b>%{y}</b><br>Avg Order Value: $%{x:,.0f}<extra></extra>",
    ))
    fig_funnel.update_layout(
        title="Avg Order Value by Customer Segment",
        paper_bgcolor="white", plot_bgcolor="#f9fafb",
        font_family="Arial", margin=dict(t=60, b=30, l=10, r=10),
        shapes=[dict(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#6da5c0", width=1.5),
            fillcolor="rgba(0,0,0,0)", layer="above",
        )],
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

# Treemap – Sales hierarchy Market → Category → Sub-Category
with nc2:
    treemap_df = df.groupby(["Market", "Category", "Sub-Category"], as_index=False)["Sales"].sum()
    fig_treemap = px.treemap(
        treemap_df,
        path=["Market", "Category", "Sub-Category"],
        values="Sales",
        color="Sales",
        color_continuous_scale=TEAL,
        title="Sales Hierarchy: Market → Category → Sub-Category",
        labels={"Sales": "Total Sales ($)"},
    )
    fig_treemap.update_traces(
        hovertemplate="<b>%{label}</b><br>Sales: $%{value:,.0f}<extra></extra>",
    )
    fig_treemap.update_layout(
        paper_bgcolor="white", font_family="Arial", margin=dict(t=60, b=10, l=10, r=10),
        shapes=[dict(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#6da5c0", width=1.5),
            fillcolor="rgba(0,0,0,0)", layer="above",
        )],
    )
    st.plotly_chart(fig_treemap, use_container_width=True)


# # ── Violin – Shipping Duration by Ship Mode (full width)
# st.markdown("<hr class='divider'>", unsafe_allow_html=True)
# st.markdown("<div class='section-title'>Shipping Efficiency</div>", unsafe_allow_html=True)

# df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days.clip(lower=0)
# fig_violin = px.violin(
#     df,
#     x="Ship Mode",
#     y="Shipping Days",
#     color="Ship Mode",
#     color_discrete_sequence=px.colors.qualitative.Vivid,
#     box=True,
#     points="outliers",
#     title="Violin – Shipping Duration by Ship Mode",
#     labels={"Shipping Days": "Days to Ship", "Ship Mode": ""},
# )
# fig_violin.update_traces(
#     hovertemplate="<b>%{x}</b><br>Days: %{y}<extra></extra>"
# )
# st.plotly_chart(style(fig_violin), use_container_width=True)


# EXPLORE SECTION – Visual Panels
if explore_option != "None":
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>🔍 Explore – {explore_option}</div>", unsafe_allow_html=True)

if explore_option == "Segment Analysis":
    ea1, ea2 = st.columns(2)

    with ea1:
        seg_sales = df.groupby("Segment", as_index=False).agg(
            Sales=("Sales","sum"), Profit=("Profit","sum"), Orders=("Order ID","nunique")
        )
        fig_seg_pie = px.pie(
            seg_sales, names="Segment", values="Sales",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hole=0.45,
            title="Sales Share by Customer Segment",
        )
        fig_seg_pie.update_traces(
            hovertemplate="<b>%{label}</b><br>Sales: $%{value:,.0f}<br>Share: %{percent}<extra></extra>"
        )
        st.plotly_chart(style(fig_seg_pie), use_container_width=True)

    with ea2:
        seg_year = df.groupby(["Year","Segment"], as_index=False)["Sales"].sum()
        fig_seg_bar = px.bar(
            seg_year, x="Year", y="Sales", color="Segment",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            barmode="group",
            title="Sales by Segment per Year",
            labels={"Sales":"Total Sales ($)"},
            text_auto=".2s",
        )
        st.plotly_chart(style(fig_seg_bar), use_container_width=True)

    seg_margin = df.groupby(["Segment","Category"], as_index=False).agg(
        Sales=("Sales","sum"), Profit=("Profit","sum")
    )
    seg_margin["Margin %"] = seg_margin["Profit"] / seg_margin["Sales"] * 100
    fig_seg_heat = px.density_heatmap(
        seg_margin, x="Segment", y="Category", z="Margin %",
        color_continuous_scale="RdYlGn",
        title="Profit Margin % – Segment × Category",
        text_auto=".1f",
    )
    st.plotly_chart(style(fig_seg_heat), use_container_width=True)

elif explore_option == "Shipping Analysis":
    if "Shipping Days" not in df.columns:
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days.clip(lower=0)

    sh1, sh2 = st.columns(2)
    with sh1:
        ship_cost = df.groupby("Ship Mode", as_index=False).agg(
            Avg_Cost=("Shipping Cost","mean"), Total_Cost=("Shipping Cost","sum"),
            Orders=("Order ID","nunique")
        )
        fig_shipcost = px.bar(
            ship_cost, x="Ship Mode", y="Avg_Cost",
            color="Ship Mode",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Avg Shipping Cost by Ship Mode",
            labels={"Avg_Cost":"Avg Cost ($)","Ship Mode":""},
            text_auto=".2s",
        )
        st.plotly_chart(style(fig_shipcost), use_container_width=True)

    with sh2:
        fig_ship_days = px.histogram(
            df, x="Shipping Days", color="Ship Mode",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            barmode="overlay", opacity=0.7, nbins=30,
            title="Distribution of Shipping Days by Ship Mode",
            labels={"Shipping Days":"Days to Ship"},
        )
        st.plotly_chart(style(fig_ship_days), use_container_width=True)

    ship_profit = df.groupby(["Ship Mode","Category"], as_index=False).agg(
        Profit=("Profit","sum"), Sales=("Sales","sum")
    )
    ship_profit["Margin %"] = ship_profit["Profit"] / ship_profit["Sales"] * 100
    fig_ship_margin = px.bar(
        ship_profit, x="Ship Mode", y="Margin %", color="Category",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        barmode="group",
        title="Profit Margin by Ship Mode & Category",
        labels={"Margin %":"Profit Margin (%)"},
        text_auto=".1f",
    )
    st.plotly_chart(style(fig_ship_margin), use_container_width=True)

elif explore_option == "Market Breakdown":
    mb1, mb2 = st.columns(2)

    with mb1:
        market_df = df.groupby("Market", as_index=False).agg(
            Sales=("Sales","sum"), Profit=("Profit","sum")
        )
        market_df["Margin %"] = market_df["Profit"] / market_df["Sales"] * 100
        fig_mkt_bar = px.bar(
            market_df.sort_values("Sales", ascending=True),
            x="Sales", y="Market", orientation="h",
            color="Margin %", color_continuous_scale="RdYlGn",
            title="Total Sales by Market (colored by Margin %)",
            labels={"Sales":"Total Sales ($)","Market":""},
            text_auto=".2s",
        )
        st.plotly_chart(style(fig_mkt_bar), use_container_width=True)

    with mb2:
        mkt_year = df.groupby(["Year","Market"], as_index=False)["Sales"].sum()
        fig_mkt_line = px.line(
            mkt_year, x="Year", y="Sales", color="Market",
            markers=True,
            title="Sales Trend by Market (Year-over-Year)",
            labels={"Sales":"Total Sales ($)","Year":"Year"},
        )
        st.plotly_chart(style(fig_mkt_line), use_container_width=True)

    mkt_cat = df.groupby(["Market","Category"], as_index=False)["Sales"].sum()
    fig_mkt_sun = px.sunburst(
        mkt_cat, path=["Market","Category"], values="Sales",
        color="Sales", color_continuous_scale=TEAL,
        title="Sunburst – Sales by Market & Category",
    )
    fig_mkt_sun.update_traces(
        hovertemplate="<b>%{label}</b><br>Sales: $%{value:,.0f}<extra></extra>"
    )
    fig_mkt_sun.update_layout(
        paper_bgcolor="white", font_family="Arial", margin=dict(t=60,b=10,l=10,r=10),
        shapes=[dict(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#6da5c0", width=1.5),
            fillcolor="rgba(0,0,0,0)", layer="above",
        )],
    )
    st.plotly_chart(fig_mkt_sun, use_container_width=True)



# FOOTER
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:12px; color:black;'>"
    "Sales & Profit Analytics • Built by Team Hammer</p>",
    unsafe_allow_html=True,
)