import streamlit as st
import os
os.environ["OMP_NUM_THREADS"]="1"
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Factory Optimization System", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

DATA_PATH="Nassau Candy Distributor.csv"

FACTORY_MAP={
    "Wonka Bar - Nutty Crunch Surprise":"Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":"Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":"Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":"Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":"Wicked Choccy's",
    "Laffy Taffy":"Sugar Shack",
    "SweeTARTS":"Sugar Shack",
    "Nerds":"Sugar Shack",
    "Fun Dip":"Sugar Shack",
    "Fizzy Lifting Drinks":"Sugar Shack",
    "Everlasting Gobstopper":"Secret Factory",
    "Hair Toffee":"The Other Factory",
    "Lickable Wallpaper":"Secret Factory",
    "Wonka Gum":"Secret Factory",
    "Kazookles":"The Other Factory"
}

FACTORY_COORDS={
    "Lot's O' Nuts":(32.881893,-111.768036),
    "Wicked Choccy's":(32.076176,-81.088371),
    "Sugar Shack":(48.11914,-96.18115),
    "Secret Factory":(41.446333,-90.565487),
    "The Other Factory":(35.1175,-89.971107)
}

NUMERICAL_COLS=['Sales','Units','Gross Profit','Cost','Order Year','Ship Year','Order Month','Ship Month','Order Weekday','Factory Avg Delay','Region Avg Delay','Product Avg Delay','Profit Margin','Cost Ratio','Units Per Sale']
CATEGORICAL_COLS=['Region','Division','Factory','Ship Mode','Country/Region','City','Product Name']

@st.cache_data
def load_data(path):
    df=pd.read_csv(path)
    df.columns=df.columns.str.strip()
    df['Ship Date']=pd.to_datetime(df['Ship Date'],format='%d-%m-%Y',errors='coerce')
    df['Order Date']=pd.to_datetime(df['Order Date'],format='%d-%m-%Y',errors='coerce')
    df=df.dropna(subset=['Order Date','Ship Date'])
    df['Delay Days']=(df['Ship Date']-df['Order Date']).dt.days
    df=df[df['Delay Days']>=0]
    Q1,Q3=df['Delay Days'].quantile(0.25),df['Delay Days'].quantile(0.75)
    IQR=Q3-Q1
    df=df[(df['Delay Days']>=Q1-1.5*IQR)&(df['Delay Days']<=Q3+1.5*IQR)]
    df['Order Year']=df['Order Date'].dt.year
    df['Ship Year']=df['Ship Date'].dt.year
    df['Order Month']=df['Order Date'].dt.month
    df['Ship Month']=df['Ship Date'].dt.month
    df['Order Weekday']=df['Order Date'].dt.weekday
    df['Factory']=df['Product Name'].map(FACTORY_MAP)
    df=df.dropna(subset=['Factory'])
    df['Factory Latitude']=df['Factory'].map(lambda f:FACTORY_COORDS[f][0])
    df['Factory Longitude']=df['Factory'].map(lambda f:FACTORY_COORDS[f][1])
    df['Factory Avg Delay']=df.groupby('Factory')['Delay Days'].transform('mean')
    df['Region Avg Delay']=df.groupby('Region')['Delay Days'].transform('mean')
    df['Product Avg Delay']=df.groupby('Product Name')['Delay Days'].transform('mean')
    df['Profit Margin']=df['Gross Profit']/df['Sales'].replace(0,np.nan)
    df['Cost Ratio']=df['Cost']/df['Sales'].replace(0,np.nan)
    df['Units Per Sale']=df['Units']/df['Sales'].replace(0,np.nan)
    df=df.replace([np.inf,-np.inf],np.nan)
    df[['Profit Margin','Cost Ratio','Units Per Sale']]=df[['Profit Margin','Cost Ratio','Units Per Sale']].fillna(0)
    return df.drop(['Row ID','Order ID','Customer ID','Product ID','Postal Code','Order Date','Ship Date'],axis=1)

@st.cache_resource
def train_models(df):
    X=df.drop('Delay Days',axis=1)
    y=df['Delay Days']
    preprocessor=ColumnTransformer([
        ('num',Pipeline([('scaler',StandardScaler())]),NUMERICAL_COLS),
        ('cat',Pipeline([('onehot',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL_COLS)
    ])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    X_train_t=preprocessor.fit_transform(X_train)
    X_test_t=preprocessor.transform(X_test)
    estimators={
        "Linear Regression":LinearRegression(),
        "Random Forest":RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=500,learning_rate=0.02,max_depth=10,min_samples_split=5,min_samples_leaf=2,subsample=0.8,random_state=42)
    }
    metrics={}
    for name,est in estimators.items():
        est.fit(X_train_t,y_train)
        pred=est.predict(X_test_t)
        metrics[name]={"MAE":mean_absolute_error(y_test,pred),"RMSE":np.sqrt(mean_squared_error(y_test,pred)),"R2":r2_score(y_test,pred)}
    best_name=max(metrics,key=lambda k:metrics[k]["R2"])
    return preprocessor,estimators,metrics,best_name

@st.cache_data
def run_clustering(df):
    route_data=df.assign(Route=df['Division']+'_'+df['Region']).groupby(['Route','Product Name']).agg({'Delay Days':'mean','Sales':'mean','Units':'mean','Gross Profit':'mean','Factory Avg Delay':'mean'}).reset_index()
    scaled=StandardScaler().fit_transform(route_data[['Sales','Units','Gross Profit','Factory Avg Delay']])
    route_data['Cluster']=KMeans(n_clusters=4,random_state=42,n_init=10).fit_predict(scaled)
    return route_data

@st.cache_data
def build_recommendations(df,_preprocessor,_model,factories):
    factory_avg_delay=df.groupby('Factory')['Delay Days'].mean().to_dict()
    feature_cols=[c for c in df.columns if c!='Delay Days']
    rows=df[feature_cols+['Delay Days']].to_dict('records')
    sim_frames=[]
    meta=[]
    for row in rows:
        current_factory=row['Factory']
        for factory in factories:
            if factory==current_factory:
                continue
            sim=dict(row)
            sim['Factory']=factory
            sim['Factory Avg Delay']=factory_avg_delay[factory]
            lat,lon=FACTORY_COORDS[factory]
            sim['Factory Latitude']=lat
            sim['Factory Longitude']=lon
            sim_frames.append({k:sim[k] for k in feature_cols})
            meta.append((row['Product Name'],current_factory,factory,row['Delay Days'],row['Gross Profit']))
    sim_df=pd.DataFrame(sim_frames)
    predicted=_model.predict(_preprocessor.transform(sim_df))
    recs=[]
    for (product,current_factory,factory,current_delay,current_profit),pred_delay in zip(meta,predicted):
        if pred_delay>=current_delay:
            continue
        lead_time_reduction=current_delay-pred_delay
        risk_reduction=lead_time_reduction/current_delay*100 if current_delay else 0
        profit_impact=lead_time_reduction*current_profit
        recs.append({'Product':product,'Current Factory':current_factory,'Recommended Factory':factory,'Current Delay':current_delay,'Predicted Delay':pred_delay,'Lead Time Reduction':lead_time_reduction,'Risk Reduction %':risk_reduction,'Profit Impact':profit_impact})
    recs_df=pd.DataFrame(recs)
    lead_norm=(recs_df['Lead Time Reduction']-recs_df['Lead Time Reduction'].min())/(recs_df['Lead Time Reduction'].max()-recs_df['Lead Time Reduction'].min())
    profit_norm=(recs_df['Profit Impact']-recs_df['Profit Impact'].min())/(recs_df['Profit Impact'].max()-recs_df['Profit Impact'].min())
    recs_df['Lead Time Norm']=lead_norm.fillna(0)
    recs_df['Profit Norm']=profit_norm.fillna(0)
    recs_df['Optimization Score']=0.5*recs_df['Lead Time Norm']+0.5*recs_df['Profit Norm']
    return recs_df.sort_values(by=['Lead Time Reduction','Risk Reduction %','Profit Impact'],ascending=False).reset_index(drop=True)

def compute_kpis(recommendations_df,df,best_r2,n_products):
    if recommendations_df.empty:
        return 0,0,0,0,0
    lead_time_reduction_pct=recommendations_df['Lead Time Reduction'].mean()/recommendations_df['Current Delay'].mean()*100
    profit_impact_stability=recommendations_df['Profit Impact'].mean()/recommendations_df['Profit Impact'].max()*100 if recommendations_df['Profit Impact'].max() else 0
    scenario_confidence=max(0,best_r2)*100
    recommendation_coverage=recommendations_df['Product'].nunique()/n_products*100
    avg_risk_reduction=recommendations_df['Risk Reduction %'].mean()
    return lead_time_reduction_pct,profit_impact_stability,scenario_confidence,recommendation_coverage,avg_risk_reduction

st.markdown("""
<style>
section.main > div { max-width: 100% !important; padding: 1.5rem 2.5rem !important; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 14px 16px; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 10px 18px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""",unsafe_allow_html=True)

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at {DATA_PATH}. Place 'Nassau Candy Distributor.csv' next to app.py.")
    st.stop()

df=load_data(DATA_PATH)
preprocessor,estimators,metrics,best_model_name=train_models(df)
model=estimators[best_model_name]
route_data=run_clustering(df)
factories=sorted(df['Factory'].unique())
recommendations_df=build_recommendations(df,preprocessor,model,factories)
lead_time_reduction_pct,profit_impact_stability,scenario_confidence,recommendation_coverage,avg_risk_reduction=compute_kpis(recommendations_df,df,metrics[best_model_name]["R2"],df['Product Name'].nunique())

st.title("Factory Reallocation & Shipping Optimization System")

m1,m2,m3,m4=st.columns(4)
m1.metric("📦 Total Orders",len(df))
m2.metric("🍬 Products",df['Product Name'].nunique())
m3.metric("🏭 Factories",df['Factory'].nunique())
m4.metric("✅ Recommendations",len(recommendations_df))

st.sidebar.header("⚙ Optimization Controls")
selected_product=st.sidebar.selectbox("Product",sorted(df['Product Name'].unique()))
selected_region=st.sidebar.selectbox("Region",sorted(df['Region'].unique()))
st.sidebar.markdown("---")
selected_shipmode=st.sidebar.selectbox("Ship Mode",sorted(df['Ship Mode'].unique()))
st.sidebar.markdown("---")
st.sidebar.caption("Slide left for speed-focused recommendations, right for profit-focused ones.")
priority=st.sidebar.slider("Optimization Priority (Speed vs Profit)",0,100,50)
speed_weight=priority/100
profit_weight=1-speed_weight

filtered_df=df[(df['Product Name']==selected_product)&(df['Region']==selected_region)&(df['Ship Mode']==selected_shipmode)]

tab_dash,tab_analytics,tab_sim,tab_rec,tab_eval=st.tabs(["📊 Dashboard","📈 Analytics","🤖 Simulator","📋 Recommendations","📉 Model Evaluation"])

with tab_dash:
    d1,d2=st.columns(2)
    with d1:
        factory_delay=df.groupby('Factory')['Delay Days'].mean().reset_index().sort_values(by='Delay Days',ascending=False)
        fig=px.bar(factory_delay,x='Factory',y='Delay Days',text='Delay Days',color='Delay Days',title="Average Shipping Delay by Factory")
        fig.update_traces(texttemplate="%{text:.2f}",textposition="outside")
        st.plotly_chart(fig,use_container_width=True)
    with d2:
        fig=px.histogram(df,x='Delay Days',nbins=20,color='Division',title="Distribution of Shipping Delays")
        st.plotly_chart(fig,use_container_width=True)

with tab_analytics:
    c1,c2=st.columns(2)
    with c1:
        fig=px.scatter(df,x='Gross Profit',y='Delay Days',color='Division',size='Sales',hover_data=['Product Name','Factory','Region'],title="Gross Profit vs Shipping Delay")
        st.plotly_chart(fig,use_container_width=True)
        fig=px.scatter(route_data,x='Sales',y='Delay Days',color=route_data['Cluster'].astype(str),size='Units',hover_data=['Route','Product Name'],title="Route Clusters by Delay & Sales")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        heatmap=df.pivot_table(values='Delay Days',index='Region',columns='Factory',aggfunc='mean')
        fig=px.imshow(heatmap,text_auto=".2f",aspect="auto",color_continuous_scale="RdYlGn_r",title="Average Delay: Region vs Factory")
        st.plotly_chart(fig,use_container_width=True)
        rec_freq=recommendations_df['Recommended Factory'].value_counts().sort_values(ascending=False).reset_index()
        rec_freq.columns=['Factory','Recommendation Count']
        fig=px.bar(rec_freq,x='Factory',y='Recommendation Count',title="Recommendation Frequency by Factory")
        st.plotly_chart(fig,use_container_width=True)

with tab_sim:
    sim_product=st.selectbox("Select Product to Simulate",sorted(df['Product Name'].unique()),key="sim_product")
    product_rows=df[df['Product Name']==sim_product]
    if product_rows.empty:
        st.warning("No records found for this product.")
    else:
        sample=product_rows.iloc[0]
        current_factory=sample['Factory']
        current_delay=sample['Delay Days']
        sim_recs=recommendations_df[recommendations_df['Product']==sim_product].sort_values('Predicted Delay')
        if sim_recs.empty:
            st.error(f"No better factory found for {sim_product}. Current factory ({current_factory}) is already optimal.")
        else:
            best=sim_recs.iloc[0]
            improvement_pct=best['Lead Time Reduction']/current_delay*100 if current_delay else 0
            s1,s2,s3,s4,s5,s6=st.columns(6)
            s1.metric("Current Factory",current_factory)
            s2.metric("Current Delay",f"{current_delay:.1f}d")
            s3.metric("Best Factory",best['Recommended Factory'])
            s4.metric("Predicted Delay",f"{best['Predicted Delay']:.1f}d")
            s5.metric("Days Saved",f"{best['Lead Time Reduction']:.1f}d")
            s6.metric("Profit Impact",f"{best['Profit Impact']:.0f}")
            if improvement_pct>=15:
                st.success(f"Reassigning {sim_product} from {current_factory} to {best['Recommended Factory']} cuts delay by {best['Lead Time Reduction']:.1f} days ({improvement_pct:.1f}%).")
            else:
                st.warning(f"Best alternative ({best['Recommended Factory']}) only improves delay by {best['Lead Time Reduction']:.1f} days ({improvement_pct:.1f}%).")
        st.dataframe(sim_recs[['Recommended Factory','Current Delay','Predicted Delay','Lead Time Reduction','Profit Impact']],use_container_width=True)
        map_df=pd.DataFrame([{'Factory':f,'Latitude':FACTORY_COORDS[f][0],'Longitude':FACTORY_COORDS[f][1],'Avg Delay':df[df['Factory']==f]['Delay Days'].mean()} for f in factories])
        fig=px.scatter_geo(map_df,lat='Latitude',lon='Longitude',text='Factory',size='Avg Delay',color='Avg Delay',scope='usa',title="Factory Locations & Average Delay")
        fig.update_layout(height=550,margin=dict(l=0,r=0,t=40,b=0),geo=dict(projection_scale=1))
        fig.update_traces(marker=dict(sizeref=map_df['Avg Delay'].max()/40,sizemin=8),textposition="top center",textfont=dict(size=13,color="#1a1a2e",family="Arial Black"))
        st.plotly_chart(fig,use_container_width=True)

with tab_rec:
    product_recs=recommendations_df[recommendations_df['Product']==selected_product].copy()
    if not product_recs.empty:
        product_recs['Optimization Score']=speed_weight*product_recs['Lead Time Norm']+profit_weight*product_recs['Profit Norm']
        product_recs=product_recs.sort_values(by='Optimization Score',ascending=False)
    if product_recs.empty:
        st.error(f"No reassignment improves delay for {selected_product}.")
    else:
        top=product_recs.iloc[0]
        st.markdown("##### 🥇 Best Recommendation")
        b1,b2,b3,b4=st.columns(4)
        b1.metric("Current Factory",top['Current Factory'])
        b2.metric("Recommended Factory",top['Recommended Factory'])
        b3.metric("Days Saved",f"{top['Lead Time Reduction']:.1f}d")
        b4.metric("Profit Impact",f"{top['Profit Impact']:.0f}")
        st.markdown("##### Top 10 Recommendations")
        st.dataframe(product_recs[['Current Factory','Recommended Factory','Current Delay','Predicted Delay','Lead Time Reduction','Risk Reduction %','Profit Impact','Optimization Score']].head(10),use_container_width=True,height=350)
    st.download_button("Download All Recommendations",recommendations_df.to_csv(index=False),file_name="Factory_Recommendations.csv",mime="text/csv")

with tab_eval:
    comparison=pd.DataFrame([{"Model":name,"MAE":vals["MAE"],"RMSE":vals["RMSE"],"R2":vals["R2"]} for name,vals in metrics.items()]).sort_values(by="R2",ascending=False).reset_index(drop=True)
    st.dataframe(comparison,use_container_width=True)
    if best_model_name in ("Random Forest","Gradient Boosting"):
        feature_names=preprocessor.get_feature_names_out()
        importances=pd.DataFrame({"Feature":feature_names,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False).head(15)
        fig=px.bar(importances,x="Importance",y="Feature",orientation="h",title=f"Top 15 Feature Importances ({best_model_name})")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig,use_container_width=True)
    k1,k2,k3,k4,k5,k6,k7=st.columns(7)
    k1.metric("Lead Time Reduction (%)",round(lead_time_reduction_pct,2))
    k2.metric("Profit Stability",round(profit_impact_stability,2))
    k3.metric("Avg Risk Reduction (%)",round(avg_risk_reduction,2))
    k4.metric("Coverage",round(recommendation_coverage,2))
    k5.metric("Scenario Confidence",round(scenario_confidence,2))
    k6.metric("Best Model",best_model_name)
    k7.metric("R2",round(metrics[best_model_name]["R2"],4))

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Factory Reallocation & Shipping Optimization Recommendation System<br>Built using Streamlit, Scikit-Learn and Plotly</p>",unsafe_allow_html=True)
