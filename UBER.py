from plotly.subplots import make_subplots
import plotly.graph_objects as go
import polars as pl
import numpy as np

data = pl.read_csv('/home/omran-xy/Workspace/Cellula/Forth task/final_internship_data.csv')

def validate_coordinates(df: pl.DataFrame, return_invalid=False):
    coord_columns = ["pickup_longitude", "pickup_latitude", 
                     "dropoff_longitude", "dropoff_latitude"]
    
    # Ensure coordinates are within a valid range (example: longitude [-180, 180], latitude [-90, 90])
    valid_coords = (
        (df["pickup_longitude"].abs() <= 180) & 
        (df["pickup_latitude"].abs() <= 90) & 
        (df["dropoff_longitude"].abs() <= 180) & 
        (df["dropoff_latitude"].abs() <= 90)
    )

    # Apply rounding to coordinate columns
    df = df.with_columns([
        pl.col(col).round(6).alias(col) for col in coord_columns
    ])

    # Split into valid and invalid DataFrames
    valid_df = df.filter(valid_coords)
    invalid_df = df.filter(~valid_coords)
    
    if return_invalid:
        return valid_df, invalid_df
    return valid_df

# Example usage
cleaned_df, invalid_df = validate_coordinates(data, return_invalid=True)
print(f"Valid rows: {cleaned_df.shape}, Invalid rows: {invalid_df.shape}")
print(invalid_df.head(10))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_ride_analysis_dashboard(df):
    """
    Creates an enhanced analysis dashboard of ride patterns using Polars DataFrame
    with improved styling and interactivity
    """
    # Create subplot layout with larger spacing and better proportions
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "surface"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]],
        subplot_titles=(
            '<b>Fare by Hour and Day</b>',
            '<b>Average Fare by Vehicle Condition</b>',
            '<b>Weather Distribution</b>',
            '<b>Distance vs Fare by Traffic</b>'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Color schemes
    surface_colors = [[0, 'rgb(17, 7, 170)'], [0.5, 'rgb(133, 8, 182)'], [1, 'rgb(255, 128, 0)']]
    bar_colors = 'rgb(255, 87, 51)'
    pie_colors = ['rgb(70, 130, 180)', 'rgb(255, 127, 80)', 'rgb(152, 251, 152)', 'rgb(238, 130, 238)']
    scatter_colors = ['rgb(65, 105, 225)', 'rgb(255, 140, 0)', 'rgb(50, 205, 50)']
    
    # 1. Enhanced 3D Surface plot
    unique_hours = sorted(df['hour'].unique().to_list())
    unique_days = sorted(df['day'].unique().to_list())
    fare_surface = np.zeros((len(unique_hours), len(unique_days)))
    
    for i, hour in enumerate(unique_hours):
        for j, day in enumerate(unique_days):
            mask = (df['hour'] == hour) & (df['day'] == day)
            fare_surface[i, j] = df.filter(mask)['fare_amount'].mean()
    
    fig.add_trace(
        go.Surface(
            z=fare_surface,
            colorscale=surface_colors,
            showscale=True,
            colorbar=dict(
                title='Fare ($)',
                x=0.45
            )
        ),
        row=1, col=1
    )
    
    # 2. Enhanced Bar chart
    unique_conditions = df['Car Condition'].unique().to_list()
    condition_fares = []
    
    for condition in unique_conditions:
        mask = df['Car Condition'] == condition
        avg_fare = df.filter(mask)['fare_amount'].mean()
        condition_fares.append({'condition': condition, 'fare': avg_fare})
    
    condition_fares.sort(key=lambda x: x['fare'])
    
    fig.add_trace(
        go.Bar(
            x=[item['condition'] for item in condition_fares],
            y=[item['fare'] for item in condition_fares],
            marker_color=bar_colors,
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            opacity=0.8,
            name='Average Fare'
        ),
        row=1, col=2
    )
    
    # 3. Enhanced Pie chart
    unique_weather = df['Weather'].unique().to_list()
    weather_counts = []
    
    for weather in unique_weather:
        count = (df['Weather'] == weather).sum()
        weather_counts.append({'weather': weather, 'count': count})
    
    fig.add_trace(
        go.Pie(
            labels=[item['weather'] for item in weather_counts],
            values=[item['count'] for item in weather_counts],
            name='Weather',
            marker_colors=pie_colors,
            textinfo='label+percent',
            hole=0.4,
            rotation=90
        ),
        row=2, col=1
    )
    
    # 4. Enhanced Scatter plot
    unique_traffic = df['Traffic Condition'].unique().to_list()
    
    for i, traffic_type in enumerate(unique_traffic):
        mask = df['Traffic Condition'] == traffic_type
        traffic_data = df.filter(mask)
        
        fig.add_trace(
            go.Scatter(
                x=traffic_data['distance'].to_list(),
                y=traffic_data['fare_amount'].to_list(),
                mode='markers',
                name=traffic_type,
                marker=dict(
                    size=6,
                    color=scatter_colors[i],
                    opacity=0.6,
                    line=dict(
                        color='white',
                        width=0.5
                    )
                )
            ),
            row=2, col=2
        )
    
    # Update layout with modern styling
    fig.update_layout(
        template='plotly_white',
        height=1000,
        width=1200,
        title=dict(
            text="<b>Ride Analysis Dashboard</b>",
            x=0.5,
            y=0.98,
            font=dict(size=24)
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        paper_bgcolor='rgb(248, 249, 250)',
        plot_bgcolor='rgb(248, 249, 250)'
    )
    
    # Update 3D surface plot
    fig.update_scenes(
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        xaxis_title="Day",
        yaxis_title="Hour",
        zaxis_title="Average Fare ($)",
        bgcolor='rgb(248, 249, 250)'
    )
    
    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0, 0, 0, 0.1)',
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0, 0, 0, 0.1)',
        zeroline=False
    )
    
    # Update specific axis labels
    fig.update_xaxes(title_text="Vehicle Condition", row=1, col=2)
    fig.update_yaxes(title_text="Average Fare ($)", row=1, col=2)
    fig.update_xaxes(title_text="Distance (miles)", row=2, col=2)
    fig.update_yaxes(title_text="Fare Amount ($)", row=2, col=2)
    
    return fig
fig=create_ride_analysis_dashboard(cleaned_df)
fig.show()