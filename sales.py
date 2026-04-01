# %%
#importing libraries

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import plotly.io as pio
import warnings

warnings.filterwarnings("ignore")
pio.renderers.default = "vscode"  # change to "vscode" if charts don't show

print("Libraries imported successfully.")


# %%
#Load dataset
FILE_PATH = "Global_Superstore2.csv"   # ← adjust path if needed

df = pd.read_csv(FILE_PATH, encoding="latin1")

print("=" * 55)
print("INITIAL DATA OVERVIEW")
print("=" * 55)
print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names   :\n{df.columns.tolist()}")
print(f"\nData types     :\n{df.dtypes}")
print(f"\nNull values    :\n{df.isnull().sum()}")
print(f"\nBasic stats    :\n{df.describe()}")
print(f"\nFirst 3 rows   :\n{df.head(3)}")

# %%
#DATA CLEANING

#Remove rows where key numeric columns are null
key_cols = ["Sales", "Quantity", "Profit", "Shipping Cost"]
rows_before = len(df)
df.dropna(subset=key_cols, inplace=True)
print(f"Rows removed (null in key cols): {rows_before - len(df)}")

#Parse mixed date formats into uniform datetime
def parse_dates(series):
    """Try multiple date formats and return a datetime Series."""
    formats = ["%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"]
    result = pd.Series([pd.NaT] * len(series), index=series.index)
    for fmt in formats:
        mask = result.isna()
        if mask.sum() == 0:
            break
        result[mask] = pd.to_datetime(series[mask], format=fmt, errors="coerce")
    #Final fallback with mixed-format inference
    still_na = result.isna()
    if still_na.sum():
        result[still_na] = pd.to_datetime(
            series[still_na], dayfirst=True, errors="coerce"
        )
    return result

df["Order Date"] = parse_dates(df["Order Date"])
df["Ship Date"]  = parse_dates(df["Ship Date"])

#Drop rows where dates couldn't be parsed
rows_before = len(df)
df.dropna(subset=["Order Date", "Ship Date"], inplace=True)
print(f"Rows removed (unparseable dates): {rows_before - len(df)}")

#Remap years: 2011→2021, 2012→2022, 2013→2023, 2014→2024, some date are written wrong
YEAR_MAP = {2011: 2021, 2012: 2022, 2013: 2023, 2014: 2024}

def remap_year(dt_series):
    """Remap years safely, handling leap day edge cases."""
    result = []
    for dt in dt_series:
        new_year = YEAR_MAP.get(dt.year, dt.year)
        try:
            result.append(dt.replace(year=new_year))
        except ValueError:
            #Feb 29 in leap year mapped to non-leap year → use Feb 28
            result.append(dt.replace(year=new_year, day=28))
    return pd.to_datetime(result)

df["Order Date"] = remap_year(df["Order Date"])
df["Ship Date"]  = remap_year(df["Ship Date"])

#Sort by Order Date ascending
df.sort_values("Order Date", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nOrder Date range: "
      f"{df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
print(f"Final shape     : {df.shape}")
print("Data cleaning complete.")

# %%
#FEATURE ENGINEERING

#Time-based columns (used in line chart and heat map)
df["Year"]  = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

#Aggregated summaries

#Total sales per year
total_sales_year = (
    df.groupby("Year", as_index=False)["Sales"]
    .sum()
    .rename(columns={"Sales": "Total_Sales"})
)

#Monthly sales
monthly_sales = df.groupby("Month", as_index=False)["Sales"].sum()
monthly_sales["Month_dt"] = pd.to_datetime(monthly_sales["Month"])
monthly_sales.sort_values("Month_dt", inplace=True)

#Sales & profit by category
sales_by_category = (
    df.groupby("Category", as_index=False)
    .agg(Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"))
)

#Sales & profit by sub-category 
sales_by_subcategory = (
    df.groupby("Sub-Category", as_index=False)
    .agg(Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"))
    .sort_values("Total_Sales", ascending=False)
)

#Sales by country
country_sales = df.groupby("Country", as_index=False)["Sales"].sum()

#Print summaries
print("Total Sales per Year:\n", total_sales_year.to_string(index=False))
print("\nSales & Profit per Category:\n", sales_by_category.to_string(index=False))
print("\nTop 5 Sub-Categories:\n", sales_by_subcategory.head(5).to_string(index=False))
print("\nFeature engineering complete.")

# %%
#Shared colour palette and styling helper (run before any chart cell)
TEAL = px.colors.sequential.Teal

def style(fig, title):
    """Apply uniform layout to every chart."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font_size=16),
        paper_bgcolor="white",
        plot_bgcolor="#f9fafb",
        font_family="Arial",
        margin=dict(t=60, b=40, l=40, r=40),
        hovermode="closest",
    )
    return fig

print("Chart styling helper ready.")

# %%
#CHART 1: LINE CHART – Monthly Sales Trend

fig_line = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    color_discrete_sequence=[TEAL[5]],
    labels={"Sales": "Total Sales ($)", "Month": "Month"},
)
fig_line.update_traces(
    hovertemplate="<b>Month:</b> %{x}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>"
)
fig_line = style(fig_line, "Monthly Sales Trend")
fig_line.show()

# %%
#CHART 2: BAR CHART – Total Sales by Category

fig_bar = px.bar(
    sales_by_category.sort_values("Total_Sales"),
    x="Total_Sales",
    y="Category",
    orientation="h",
    color="Total_Sales",
    color_continuous_scale=TEAL,
    labels={"Total_Sales": "Total Sales ($)", "Category": ""},
    text_auto=".2s",
)
fig_bar.update_traces(
    hovertemplate="<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>"
)
fig_bar = style(fig_bar, "Total Sales by Category")
fig_bar.show()

# %%
#CHART 3: HEAT MAP – Sales by Category × Year

pivot = df.pivot_table(
    index="Category",
    columns="Year",
    values="Sales",
    aggfunc="sum"
)

fig_heat = px.imshow(
    pivot,
    color_continuous_scale="RdYlGn",
    text_auto=".2s",
    aspect="auto",
    labels=dict(x="Year", y="Category", color="Sales ($)"),
)
fig_heat.update_traces(
    hovertemplate=(
        "<b>Category:</b> %{y}<br>"
        "<b>Year:</b> %{x}<br>"
        "<b>Sales:</b> $%{z:,.0f}<extra></extra>"
    )
)
fig_heat = style(fig_heat, "🔥 Heat Map – Sales by Category & Year")
fig_heat.show()

# %%
#CHART 4: BOX PLOT – Profit Distribution by Category

# Safely pick 3 colours regardless of palette length
n = len(TEAL)
c1 = TEAL[int(n * 0.3)]   # ~30% through the palette
c2 = TEAL[int(n * 0.6)]   # ~60% through the palette
c3 = TEAL[int(n * 0.9)]   # ~90% through the palette

fig_box = px.box(
    df,
    x="Category",
    y="Profit",
    color="Category",
    color_discrete_sequence=[c1, c2, c3],
    points="outliers",
    labels={"Profit": "Profit ($)", "Category": ""},
)
fig_box.update_traces(
    hovertemplate=(
        "<b>Category:</b> %{x}<br>"
        "<b>Profit:</b> $%{y:,.0f}<extra></extra>"
    )
)
fig_box = style(fig_box, "📦 Profit Distribution by Category")
fig_box.show()

# %%
# CHART 5: SCATTER PLOT – Sales vs Profit by Category (colorful)

fig_scatter = px.scatter(
    df.sample(n=min(2000, len(df)), random_state=42),
    x="Sales",
    y="Profit",
    color="Category",                        # colour by Category
    color_discrete_sequence=px.colors.qualitative.Vivid,  # colorful distinct palette
    size="Sales",
    size_max=15,
    opacity=0.7,
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
fig_scatter = style(fig_scatter, "Scatter Plot – Sales vs Profit by Category")
fig_scatter.show()

# %%
#CHART 6: HISTOGRAM + DENSITY – Multi-layered Chart

fig_hist_kde = go.Figure()

#Safely pick colours based on actual palette length
n  = len(TEAL)
c1 = TEAL[int(n * 0.4)]   # histogram bar colour
c2 = TEAL[int(n * 0.7)]   # sales KDE line colour
c3 = TEAL[int(n * 0.9)]   # profit KDE line colour

#Calculate sensible x-axis limits using percentiles (removes extreme outliers)
sales_p01  = df["Sales"].quantile(0.01)    # 1st  percentile
sales_p99  = df["Sales"].quantile(0.99)    # 99th percentile
profit_p01 = df["Profit"].quantile(0.01)
profit_p99 = df["Profit"].quantile(0.99)

#Overall x range covering both Sales and Profit (clipped)
x_min = min(sales_p01, profit_p01)
x_max = max(sales_p99, profit_p99)

print(f"Sales  range (1st–99th pct): ${sales_p01:,.0f}  →  ${sales_p99:,.0f}")
print(f"Profit range (1st–99th pct): ${profit_p01:,.0f}  →  ${profit_p99:,.0f}")
print(f"X-axis will show           : ${x_min:,.0f}  →  ${x_max:,.0f}")

#Filter data to clipped range for cleaner visualisation
sales_clipped  = df["Sales"][(df["Sales"]  >= sales_p01)  & (df["Sales"]  <= sales_p99)]
profit_clipped = df["Profit"][(df["Profit"] >= profit_p01) & (df["Profit"] <= profit_p99)]

#Layer 1: Histogram of Sales (left y-axis)
fig_hist_kde.add_trace(
    go.Histogram(
        x=sales_clipped,
        nbinsx=60,
        name="Sales Count",
        marker=dict(
            color=c1,                              # bin fill colour
            line=dict(color="white", width=0.8)    # ← border added here
        ),
        opacity=0.6,
        yaxis="y1",
        hovertemplate="Sales: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    )
)

#Layer 2: KDE curve for Sales (right y-axis)
kde_sales   = gaussian_kde(sales_clipped, bw_method=0.15)
x_sales     = np.linspace(sales_p01, sales_p99, 500)
y_sales_kde = kde_sales(x_sales)

fig_hist_kde.add_trace(
    go.Scatter(
        x=x_sales,
        y=y_sales_kde,
        name="Sales Density (KDE)",
        line=dict(color=c2, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56,166,165,0.12)",
        yaxis="y2",
        hovertemplate="Sales: $%{x:,.0f}<br>Density: %{y:.6f}<extra></extra>",
    )
)

#Layer 3: KDE curve for Profit (right y-axis)
kde_profit   = gaussian_kde(profit_clipped, bw_method=0.15)
x_profit     = np.linspace(profit_p01, profit_p99, 500)
y_profit_kde = kde_profit(x_profit)

fig_hist_kde.add_trace(
    go.Scatter(
        x=x_profit,
        y=y_profit_kde,
        name="Profit Density (KDE)",
        line=dict(color=c3, width=2.5, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(8,104,172,0.10)",
        yaxis="y2",
        hovertemplate="Value: $%{x:,.0f}<br>Density: %{y:.6f}<extra></extra>",
    )
)

#Dual y-axis layout
fig_hist_kde.update_layout(
    title=dict(
        text="Histogram + Density – Sales & Profit Distribution",
        x=0.5, xanchor="center", font_size=16
    ),
    xaxis=dict(
        title="Value ($)",
        range=[x_min, x_max],      # ← clipped range so bins are visible
        tickprefix="$",
        tickformat=",.0f",
    ),
    yaxis=dict(
        title=dict(
            text="Order Count",
            font=dict(color=c1)
        ),
        tickfont=dict(color=c1),
    ),
    yaxis2=dict(
        title=dict(
            text="Density",
            font=dict(color=c2)
        ),
        tickfont=dict(color=c2),
        overlaying="y",
        side="right",
        showgrid=False,
    ),
    legend=dict(x=0.75, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    paper_bgcolor="white",
    plot_bgcolor="#f9fafb",
    font_family="Arial",
    hovermode="x unified",
    margin=dict(t=60, b=40, l=40, r=60),
    barmode="overlay",
)
fig_hist_kde.show()

# %%
#CHART 7: BUBBLE CHART – Sub-Category Sales vs Profit

bubble_df = sales_by_subcategory.copy()
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
    labels={
        "Total_Sales": "Total Sales ($)",
        "Total_Profit": "Total Profit ($)"
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
fig_bubble = style(fig_bubble, "Bubble Chart – Sub-Category Sales vs Profit")
fig_bubble.show()

# %%
# ── CHART 8: GEOGRAPHICAL MAP – Total Sales by Country

fig_map = px.choropleth(
    country_sales,
    locations="Country",
    locationmode="country names",
    color="Sales",
    color_continuous_scale=TEAL,
    projection="natural earth",
    labels={"Sales": "Total Sales ($)"},
)

# ── Add latitude lines (parallels)
for lat in range(-90, 91, 30):
    fig_map.add_trace(
        go.Scattergeo(
            lat=[lat] * 361,
            lon=list(range(-180, 181)),
            mode="lines",
            line=dict(color="rgba(150,150,150,0.4)", width=0.6),
            showlegend=False,
            hoverinfo="skip",
            name=f"{lat}° Lat",
        )
    )

# ── Add longitude lines (meridians)
for lon in range(-180, 181, 30):
    fig_map.add_trace(
        go.Scattergeo(
            lat=list(range(-90, 91)),
            lon=[lon] * 181,
            mode="lines",
            line=dict(color="rgba(150,150,150,0.4)", width=0.6),
            showlegend=False,
            hoverinfo="skip",
            name=f"{lon}° Lon",
        )
    )

# ── Update map styling (fixed: showborders → showcountries & countrycolor)
fig_map.update_geos(
    showcoastlines=True,
    coastlinecolor="white",
    coastlinewidth=0.8,
    showcountries=True,        # ← fixed (was showborders)
    countrycolor="white",      # ← fixed (was bordercolor)
    countrywidth=0.5,          # ← fixed (was borderwidth)
    showland=True,
    landcolor="#f0f0f0",
    showocean=True,
    oceancolor="#d6eaf8",
    showlakes=True,
    lakecolor="#d6eaf8",
    showrivers=False,
    lataxis=dict(
        showgrid=True,
        gridcolor="rgba(150,150,150,0.3)",
        gridwidth=0.5,
        dtick=30,
    ),
    lonaxis=dict(
        showgrid=True,
        gridcolor="rgba(150,150,150,0.3)",
        gridwidth=0.5,
        dtick=30,
    ),
)

fig_map.update_traces(
    hovertemplate="<b>%{location}</b><br>Sales: $%{z:,.0f}<extra></extra>",
    selector=dict(type="choropleth"),
)

fig_map = style(fig_map, "Global Sales by Country")
fig_map.show()

# %%



