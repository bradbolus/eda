import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly as py
import seaborn as sns
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import sweetviz as sv
import streamlit.components.v1 as components
import codecs
from streamlit_plotly_events import plotly_events


# Create a side menu 
menu = ["EDA APP", "Custom Dataframe Selection", "Customer Segmentation"]
choice = st.sidebar.selectbox('What would you like to try first?', menu)
# Create the Home page
if choice == "EDA APP":    

    # Web App Title
    st.markdown('''
    # **Exploratory Data Analysis**
    This is an **EDA App** created in Streamlit using the **Pandas-Profiling** library and **HAB LABS** expertise.
    ''')

    st.image("4445(1).jpg")

    st.write("📊 We enable your business to combine multiple sources of data into one reliable, user-friendly location.")
    st.write("📊 Try it out on your own dataset by uploading it ⬅︎ or have a look at our example of a **Mall Dataset**")
    # Upload CSV data
    with st.sidebar.header('Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file here", type=['csv'])


    # Pandas Profiling Report
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        df = load_csv()
        columns = df.columns
        df_cat = df.select_dtypes(exclude=["number","bool_","object_"])
        df_num = df.select_dtypes(include=["number","bool_","object_"])
        
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Overview of your Data**')

        #-- Analyse with Sweetviz
        def st_display_sweetviz(report_html, width = 1000, height= 500):
            report_file = codecs.open(report_html, "r")
            page = report_file.read()
            components.html(page, width = width, height = height,scrolling = True)
        
        if st.button("Generate Sweetviz Report"):
            report = sv.analyze(df)
            report.show_html()
            #st_display_sweetviz("SWEETVIZ_REPORT.html")

        st.markdown('''
        # **FREE CONSULTATION**
        If you like what you see dont hesitate to contact us for a chat ⬇︎!
        ''')
        st.button("FREE CONSULTATION")    
        
        

    else:
        st.info('Awaiting a CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Example data
            @st.cache(allow_output_mutation=True)
            def load_data():
                a = pd.read_csv("Mall_Customers.csv")
                return a
            df = load_data()
            pr = ProfileReport(df, explorative=True)
            st.write("---")
            st.header('**Input DataFrame**')
            st.write("Take a look at the dataset we will be working with below ⬇︎")
            st.write(df)
            
            st.write('---')
            
            st.header('**Pandas Profiling Report**')
            st.write("📊 The first part of our data analysis of this dataset comes in the form or a **Pandas Profiling Report**")
            st.write("📊 You can think of this as a nuts and bolts, deepdive into your data. It focusses on:") 
            st.write(" 1) **Numerical Breakdowns** of each column of your data")
            st.write(" 2) **Missing Values** -- Missing datapoints or entries in your dataset")
            st.write(" 3) **Correlation between features** -- How interdependant two or more columns are in your dataset")
            st.write(" 4) **Cardinality of features** -- How unique each entry in specific columns are. Low value means all of the entries are the same")
            st.write(" 5) **Interactions between features** -- How do your columns relate to one another")
            st.write("📊  Feel free to play around with the report below!")
            
            my_expander = st.expander(label='Open Pandas Profiling Report Here')
            with my_expander:
                st_profile_report(pr)

            st.write("---")

            st.header("Next we focus on **VISUALIZING** the insights we have found in your data")
            
            # --- PLOT PIE CHART
            gender_count = df["Gender"].value_counts().values
            labels = ['Female', 'Male']
            explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

            fig1, ax1 = plt.subplots()
            plt.title("Gender Profile of Customers")
            ax1.pie(gender_count, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)

            st.write("📊  The majority of the shoppers in our dataset are **female**")

            st.write("---")
            #- Table explaining age groupbrekadown
            st.header("Description of Age Groups Breakdown")
            st.markdown(
                """
            | Feature | Description |
            | --- | --- |
            | `Teenager` | Shoppers YOUNGER than 21 years old |
            | `Young Adult` | Shoppers between the age of 21 and 30 |
            | `Adult` | Shoppers between the age of 30 and 40 |
            | `Middle Aged` | Shoppers between the age of 40 and 60|
            | `Senior Citizen` | Shoppers OLDER than 60 years old |
             
            """
            )
            st.write ("---")
            
            
            #---- Two bar plot Age Gender
            # Create new column for age group
            
            df2 = df.copy()
            df2.loc[(df2.Age < 21), 'AgeGroup'] = 'Teenager'
            df2.loc[(df2.Age >= 21) & (df2.Age <= 30),  'AgeGroup'] = 'Young Adult'
            df2.loc[(df2.Age > 30) & (df2.Age <= 40), 'AgeGroup'] = 'Adult'
            df2.loc[(df2.Age > 40) & (df2.Age <= 60),  'AgeGroup'] = 'Middle Aged'
            df2.loc[(df2.Age > 60), 'AgeGroup'] = 'Senior Citizen'

            
            
            # Plotting AGE GENDER
            gender_age = df2.groupby(['Gender', "AgeGroup"]).size().reset_index(name="Count")

            fig2, ax2 = plt.subplots(figsize=(12, 10))

            plt.title("Gender and Age Group Profile of Customers", size = 20)
            g = sns.barplot(x='Gender', y='Count', hue='AgeGroup', data=gender_age, palette = "mako", ax=ax2)

            plt.ylabel("Count", size=20)
            plt.xlabel("AgeGroup", size=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize = 18)
            plt.grid(False)
            plt.legend(fontsize=20)

            for p in g.patches:
                g.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                size=15,
                xytext = (0, 10), 
                textcoords = 'offset points')
            sns.despine(fig2)
            st.pyplot(fig2)

            st.write("📊 ⬆︎ Is a count of each of defined age groups by gender")
            st.write("📊  We dive into this a little more in depth ⬇︎")

            st.write("---")

            #Plot Proprotion of Each Age Bracket and Gender

            a = gender_age[gender_age["Gender"] == "Female"]
            #a.drop(columns ="AgeGroup")
            a = a["Count"]

            b = gender_age[gender_age["Gender"] == "Male"]
            #b.drop(columns ="AgeGroup")
            b = b["Count"]

            mylabels = ["Adult", "Middle Aged", "Senior Citizen", "Teenager", "Young Adult"]
            myexplode = [0.05, 0, 0 , 0, 0]
            mycolors = ["lightblue", "pink", "yellow", "orange", "red"]

            fig5, ax5 = plt.subplots(figsize=(15, 10))
            
            plt.title("Female: Age Group Breakdown", fontsize=20)
            plt.pie(a, autopct='%.1f%%', labels = mylabels, explode = [0, 0.05,0,0,0], colors = mycolors, textprops={'fontsize': 16})
            st.pyplot(fig5)

            st.write("📊  Our female customers are largely made up of customers in our **Young Adult**,**Adult** and **Middle Aged** age groups")

            fig6, ax6 = plt.subplots(figsize=(15, 10))
            plt.title("Male: Age Group Breakdown ", fontsize=20)
            plt.pie(b, autopct='%.1f%%', labels = mylabels, explode = myexplode, colors = mycolors, textprops={'fontsize': 16})

            st.pyplot(fig6)

            st.write("📊  Male customers are devided more evenly between the age groups with the highest percentage of customers coming from the **Adult** and **Middle Aged** age groups")
            
            st.write("---")

            #AgeGroup and Spending score Distribution
            df.loc[(df.Age < 21), 'AgeGroup'] = 'Teenager'
            df.loc[(df.Age >= 21) & (df.Age <= 30),  'AgeGroup'] = 'Young Adult'
            df.loc[(df.Age > 30) & (df.Age <= 40), 'AgeGroup'] = 'Adult'
            df.loc[(df.Age > 40) & (df.Age <= 60),  'AgeGroup'] = 'Middle Aged'
            df.loc[(df.Age > 60), 'AgeGroup'] = 'Senior Citizen'

            
            df3 = df.copy()
            df3 = df3[df3["AgeGroup"]!= "Senior Citizen"]

            fig = plt.figure(figsize=(25, 20))
            gs = fig.add_gridspec(4,1)
            gs.update(hspace= -0.55)

            axes = list()
            colors = ["#004c70", "#990000",'#990000', '#990000']

            for idx, cls, c in zip(range(6), df3['AgeGroup'].unique(), colors):
                axes.append(fig.add_subplot(gs[idx, 0]))
                
                # you can also draw density plot with matplotlib + scipy.
                sns.kdeplot(x='Spending Score (1-100)', data=df3[df3['AgeGroup']==cls], 
                            fill=True, ax=axes[idx], cut=5, bw_method=0.25, 
                            lw=1.4, edgecolor='lightgray',multiple="stack", palette=['coral','cadetblue'], alpha=0.7,hue='Gender') 
                
                axes[idx].set_ylim(0, 0.04)
                axes[idx].set_xlim(0, 100)
                
                axes[idx].set_yticks([])
                if idx != 3 : axes[idx].set_xticks([])
                axes[idx].set_ylabel('')
                axes[idx].set_xlabel('')
                plt.xticks(fontsize=26)
                
                spines = ["top","right","left","bottom"]
                for s in spines:
                    axes[idx].spines[s].set_visible(False)
                    
                axes[idx].patch.set_alpha(0)
                axes[idx].text(-0.2,0.001,f'{cls} ',fontweight="light", fontfamily='sans-serif', fontsize=28,ha="right")
                if idx != 6 : axes[idx].get_legend().remove() 
                

            fig.text(0.13,0.8,"Spending Score Distribution: Gender and Age Range", fontweight="bold", fontfamily='sans-serif', fontsize=32)
            fig.text(0.13,0.77,'Interestingly, Seniors have no higher spending scores than 60.',fontfamily='sans-serif',fontsize=26)
            fig.text(0.13,0.75,'This age group has been left out of this analysis.',fontfamily='sans-serif',fontsize=26)

            fig.text(0.770,0.77,"  Male", fontweight="bold", fontfamily='sans-serif', fontsize=28, color='cadetblue')
            fig.text(0.825,0.77,"|", fontweight="bold", fontfamily='sans-serif', fontsize=28, color='black')
            fig.text(0.835,0.77,"Female", fontweight="bold", fontfamily='sans-serif', fontsize=28, color='coral')

            st.pyplot(fig) 

            st.write("📊  Next, we investgated the distribution of **spending scores** by **gender** and **age group**")
            st.write("📊  One interesting insight is that **young adults** ,for both genders, seem to have the **highest spending scores** by distribution")
            st.write("📊  Another is that **females** dominate across the board in terms of spending scores")
            st.write("📊  Finally, the two peaks of **teenager** and **middle aged** spending scores seem to mirror each other.")
            st.write("Feel free to enlarge **⤢** this graphic for a closer look! ")

            st.write("---")  

            # --- PLOT Scatter Plots with Line of Best Fit
            st.header("Age vs Spending Score and Gender")
            fig = sns.lmplot(data=df, x="Spending Score (1-100)", y="Age", hue="Gender");
            st.pyplot(fig)

            st.write("📊 For both men and women our analysis shows that age is **negatively correlated** with **spending score**")
            st.write("📊  This mean that as age ⬆︎, all things being equal, spending score⬇︎")
            st.write("📊 All things being equal, customers aged between **20** and **40** look to have **higher spending scores** at this mall")

            st.header("Annual Income vs Age and Gender")
            fig = sns.lmplot(x="Annual Income (k$)", y="Age", hue="Gender", data=df)
            st.pyplot(fig)
            
            st.write("📊 In this dataset **males** age looks to be **negatively correlated** with **annual salary** while **females** age look to be **positively correlated**")
            
            st.header("Annual Income vs Spending Score and Gender")
            fig = sns.lmplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", data=df)
            st.pyplot(fig)
            
            st.write("📊 There is **no clear** correlation between **annual income** and **spending score** by gender")

            st.write("---")

            st.subheader("For further insights check out our **CUSTOM DATAFRAME SELECTION** and **CUSTOMER SEGMENTATION** tabs ⬅︎")

            st.write("---")

            st.markdown('''
            # **FREE CONSULTATION** 

        If you like what you see dont hesitate to contact us for a chat ⬇︎!
        ''')
            st.button("FREE CONSULTATION")

            
elif choice == "Custom Dataframe Selection":

        st.markdown('''
    # **Customer Dataframe Selection**
    ''')
        st.image("19201(1).jpg")
        st.write("📊 Below we give you the ability to quickly and effeciently filter large datasets")
        st.write("📊 Additionally, we provide you with a quick but succinct overview of this fitered dataset in the form of a **Sweetviz** report")
     
    # Example data
        @st.cache(allow_output_mutation=True)

        def load_data():
            a = pd.read_csv("Mall_Customers.csv")
            return a
        df = load_data()      
           
           # --- STREAMLIT SELECTION
        
        customer_selection = st.form("Customer Selection")
        with customer_selection:

            st.header("Filter your Dataframe below ⬇︎")
            option = st.selectbox('What gender is your customer?',("Awaiting choice",'Male', 'Female'))
            st.write('You selected:', option)

            if option == "Male":
                df = df[df["Gender"]== "Male"]
                

            elif option == "Female":
                df = df[df["Gender"]== "Female"]
                
            else:
                pass    
            
            st.write("Now, drag the sliders to select a subset of your dataset")
            ages = df['Age'].unique().tolist()
            income = df['Annual Income (k$)'].unique().tolist()
            spending_score = df['Spending Score (1-100)'].unique().tolist()

            age_selection = st.slider('Age:',
                                min_value= min(ages),
                                max_value= max(ages),
                                value=(min(ages),max(ages)))

            annual_income_selection = st.slider('Annual Income:',
                                min_value= min(income),
                                max_value= max(income),
                                value=(min(income),max(income)))

            spending_score_selection = st.slider('Spending Score:',
                                min_value= min(spending_score),
                                max_value= max(spending_score),
                                value=(min(spending_score),max(spending_score)))
            
            
            submitted = st.form_submit_button("Click here to try out Custom Dataframe Selection")
            
            if submitted:
                st.subheader("Filtered Dataframe") 
                df = df[(df["Age"] >= age_selection[0]) & (df['Age'] <= age_selection[1])]
                df = df[(df["Annual Income (k$)"] >= annual_income_selection[0]) & (df['Annual Income (k$)'] <= annual_income_selection[1])]
                df = df[(df["Spending Score (1-100)"] >= spending_score_selection[0]) & (df['Spending Score (1-100)'] <= spending_score_selection[1])]
                df = df[df["Gender"] == option]
                df
                st.write("📊 Take a look at the Sweetviz report on your filtered dataset in the new tab by hitting the button below ⬇︎!")
                #-- Analyse with Sweetviz
                def st_display_sweetviz(report_html, width = 1000, height= 500):
                    report_file = codecs.open(report_html, "r")
                    page = report_file.read()
                    components.html(page, width = width, height = height,scrolling = True)
                        
        if st.button("Sweetviz Report"):
            report = sv.analyze(df)
            report.show_html()
                #st_display_sweetviz("SWEETVIZ_REPORT.html")

            st.markdown('''
        # **FREE CONSULTATION**
        If you like what you see dont hesitate to contact us for a chat ⬇︎!
        ''')
            st.button("FREE CONSULTATION")
                
        
        

elif choice == "Customer Segmentation":
    
    st.markdown('''
    # **Customer Segmentation**
    **Customer Segmentation** is the process of division of a customer base into several groups.
    ''')
    
    st.image("too-broad-customer-segmentation.jpeg")
    st.write("📊  These groups share similarities that are relevant to marketing such as gender, age, annual income and spending habits.")
    st.write("📊  Once your company understands the characteristics of these 'clusters' of clients you can divert your ad budget away from those who are unlikely to purchase your product or service towards your most valuable customers")
    st.write("📊  This customer segmentation will be completed on our **Mall Dataset**")
    
    if st.button('Press me for Customer Segmentation'):
        @st.cache(allow_output_mutation=True)
        def load_data():
            a = pd.read_csv("Mall_Customers.csv")
            return a
        df = load_data() 
        
        #-- Preparing KMeans Elbow
        X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
        inertia = []
        for n in range(1 , 11):
            algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
            algorithm.fit(X1)
            inertia.append(algorithm.inertia_)

        #-- Plotting Kmeans Elbow
        st.subheader("KMeans Elbow")
        fig, ax = plt.subplots()
        plt.plot(np.arange(1 , 11) , inertia , 'o')
        plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
        plt.xlabel("Optimal Number of Clusters")
        plt.ylabel("Distortion Score")
        st.pyplot(fig)

        st.write("📊 The first step in any customer segmentation is to work out the optimal number of groups of customers or 'clusters'") 
        st.write("📊 Following the implementation of a KMeans algorithm, the above graph shows us that in this case the optimal number of clusters is **SIX**")

        st.write("---")
        #----Creating Segemenation Illustration
        algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
        algorithm.fit(X1)
        labels1 = algorithm.labels_
        centroids1 = algorithm.cluster_centers_

        h = 0.02
        x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
        y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

        st.header("Customer Segmentation: Age and Spending Score")
        fig, ax = plt.subplots()
        plt.clf()
        Z = Z.reshape(xx.shape)
        plt.imshow(Z , interpolation='nearest', 
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

        plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 
                s = 100 )
        plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 200 , c = 'red' , alpha = 0.7)
        plt.xlabel("Age")
        plt.ylabel("Spending Score (1-100)")
        st.pyplot(fig)

        st.write(" 📊  Machine learning models are powerful decision-making tools. They can precisely identify customer segments, which is much harder to do manually or with conventional analytical methods.")
        st.write("📊  Above ⬆︎ we can see a visual representation of a customer segmentation on our **Mall Dataset**")
        st.write("📊  In this case the clusters have been segmented based on Age and Spending Score")

        st.header("Customer Segmentation: Age and Annual Income")
        fig, ax = plt.subplots()
        plt.clf()
        Z = Z.reshape(xx.shape)
        plt.imshow(Z , interpolation='nearest', 
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

        plt.scatter( x = 'Age' ,y = 'Annual Income (k$)' , data = df , c = labels1 , 
                s = 100 )
        plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 200 , c = 'red' , alpha = 0.7)
        plt.xlabel("Age")
        plt.ylabel("Annual Income(k$)")
        st.pyplot(fig)

        st.write("📊 Above ⬆︎ our clusters have been segmented based on Age and Annual Income")
        
        ##### 3D Vis
        X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
        inertia = []
        for n in range(1 , 11):
            algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                                tol=0.0001,  random_state= 111  , algorithm='elkan') )
            algorithm.fit(X3)
            inertia.append(algorithm.inertia_)

        algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                                tol=0.0001,  random_state= 111  , algorithm='elkan') )
        algorithm.fit(X3)
        labels3 = algorithm.labels_
        centroids3 = algorithm.cluster_centers_
                
        st.subheader("3D Customer Segmentation")
        st.write("📊  Feel free to play around with our 3D segmentation. If its a little confusing dont worry we provide further insights below!")
        df['label3'] =  labels3
        trace1 = go.Scatter3d(
                    x= df['Age'],
                    y= df['Spending Score (1-100)'],
                    z= df['Annual Income (k$)'],
                    mode='markers',
                    marker=dict(
                        color = df["label3"], 
                        size= 15,
                        line=dict(
                            color= df['label3'],
                            width= 12,
                        ),
                        opacity=0.7
                    )
                )
        data = [trace1]
        layout = go.Layout(
                    margin=dict(
                    l=1,
                    r=1,
                    b=1,
                    t=1
                    ),
                    scene = dict(
                            xaxis = dict(title  = 'Age'),
                            yaxis = dict(title  = 'Spending Score'),
                            zaxis = dict(title  = 'Annual Income')
                        )
                        )
        fig = go.Figure(data=data, layout=layout)
        plotly_events(fig, click_event=False, hover_event=False)
        #New 3D Customer Segmentation

        st.write("---")

        ####---- Extra work to make the below work!
        st.subheader("Customer Segmentation Insights")
        st.write("📊  For the following graphics please press the ⤢ button for a better view!")
        X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
        inertia = []
        for n in range(1 , 11):
            algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
            algorithm.fit(X3)
            inertia.append(algorithm.inertia_)

        algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
        algorithm.fit(X3)
        labels3 = algorithm.labels_
        centroids3 = algorithm.cluster_centers_
            
        df['label3'] =  labels3
        
        
        df4 = df.copy()
        df4.rename(columns ={"label3":"Cluster"}, inplace = True)

        grouped_km = df4.groupby(['Cluster']).mean().round(1)
        grouped_km2 = df4.groupby(['Cluster']).mean().round(1).reset_index()
        grouped_km2['Cluster'] = grouped_km2['Cluster'].map(str)
        grouped_km2.drop(columns =["CustomerID"], inplace = True)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grouped_km2["Spending Score (1-100)"], grouped_km2["Annual Income (k$)"], grouped_km2["Age"],color=['yellow','red','green','orange','blue','purple'],alpha=0.5,s=500)

        # add annotations one by one with a loop
        for line in range(0,grouped_km.shape[0]):
            ax.text(grouped_km2['Spending Score (1-100)'][line], grouped_km2['Annual Income (k$)'][line],grouped_km2['Age'][line], s=('Cluster \n'+grouped_km2['Cluster'][line]), horizontalalignment='center', fontsize=12, fontweight='light', fontfamily='serif')
                
        ax.set_xlabel("Spending Score (1-100)", fontsize = 12)
        ax.set_ylabel("Annual Income (k$)",fontsize = 12)
        ax.set_zlabel("Age", fontsize = 12)

        fig.text(0.15, .95, '3D Plot: Clusters Visualized', fontsize=20, fontweight='bold', fontfamily='sans-serif')
        fig.text(0.15, .9, 'Clusters by averages in 3D.', fontsize=15, fontweight='light', fontfamily='sans-serif')

        fig.text(1.172, 0.95, 'Insight', fontsize=20, fontweight='bold', fontfamily='sans-serif')

        fig.text(1.172, 0.3, '''
        We observe a clear distinction between clusters. 

        As a business, we might want to rename our clusters
        so that they have a clear & obvious meaning; right now
        the cluster labels mean nothing. 

        Let's change that:

        Cluster 0 - Middle spending score, Middle income, High age - Valuable

        Cluster 1 - High spending score, High income, Young age - Most Valuable

        Cluster 2 - Lowest spending score, High income, High age - Less Valuable

        Cluster 3 - High spending score, Low income, Young age - Very Valuable.

        Cluster 4 - Low spending score, Low income, High age - Least Valuable

        Cluster 5 - Middle spending score, Middle income, Young age - Targets.
        '''
                , fontsize=20, fontweight='light', fontfamily='sans-serif')

        import matplotlib.lines as lines
        l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
        fig.lines.extend([l1])
        st.pyplot(fig)

        st.write("---")

        #Percentages BarPlot by Gender

        df4['Cluster_Label'] = df4['Cluster'].apply(lambda x: 'Less Valuable' if x == 0 else 
                                               'Targets' if x == 1 else
                                               'Valuable' if x == 2 else
                                               'Most Valuable' if x == 3 else
                                               'Least Valuable' if x == 4 else 'Very Valuable')

        # New column for radar plots a bit later on 

        df4['Sex (100=Male)'] = df4['Gender'].apply(lambda x: 100 if x == 'Male' else 0)

        df4['Cluster'] = df4['Cluster'].map(str)
        # Order for plotting categorical vars
        Cluster_ord = ['0','1','2','3','4','5']
        clus_label_order = ['Targets','Most Valuable','Very Valuable','Valuable','Less Valuable','Least Valuable']
        

        clus_ord = df4['Cluster_Label'].value_counts().index

        clu_data = df4['Cluster_Label'].value_counts()[clus_label_order]
        ##

        data_cg = df4.groupby('Cluster_Label')['Gender'].value_counts().unstack().loc[clus_label_order]
        data_cg['sum'] = data_cg.sum(axis=1)

        ##
        data_cg_ratio = (data_cg.T / data_cg['sum']).T[['Male', 'Female']][::-1]

        
        ### Plotting
        fig, ax = plt.subplots(1,1,figsize=(18, 10))

        ax.barh(data_cg_ratio.index, data_cg_ratio['Male'], 
                color='cadetblue', alpha=0.7, label='Male')
        ax.barh(data_cg_ratio.index, data_cg_ratio['Female'], left=data_cg_ratio['Male'], 
                color='coral', alpha=0.7, label='Female')


        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticklabels((data_cg_ratio.index), fontfamily='sans-serif', fontsize=14)


        # male percentage
        for i in data_cg_ratio.index:
            ax.annotate(f"{data_cg_ratio['Male'][i]*100:.3}%", 
                        xy=(data_cg_ratio['Male'][i]/2, i),
                        va = 'center', ha='center',fontsize=14, fontweight='light', fontfamily='sans-serif',
                        color='white')

        for i in data_cg_ratio.index:
            ax.annotate(f"{data_cg_ratio['Female'][i]*100:.3}%", 
                        xy=(data_cg_ratio['Male'][i]+data_cg_ratio['Female'][i]/2, i),
                        va = 'center', ha='center',fontsize=14, fontweight='light', fontfamily='sans-serif',
                        color='#244247')
            

        fig.text(0.129, 0.98, 'Gender Distribution by Cluster', fontsize=20, fontweight='bold', fontfamily='serif')   
        fig.text(0.129, 0.88, 
                '''
        We see that females dominate most of our categories; except our Targets cluster.
        How might we encourage more male customers?
        Incentive programs for females in our Targets cluster?''' , fontsize=14,fontfamily='serif')   

        for s in ['top', 'left', 'right', 'bottom']:
            ax.spines[s].set_visible(False)
            
        ax.legend().set_visible(False)

        fig.text(0.777,0.98,"Male", fontweight="bold", fontfamily='serif', fontsize=18, color='cadetblue')
        fig.text(0.819,0.98,"|", fontweight="bold", fontfamily='serif', fontsize=18, color='black')
        fig.text(0.827,0.98,"Female", fontweight="bold", fontfamily='serif', fontsize=18, color='coral')
        st.pyplot(fig)

        st.write("---")

        st.image("Heatmap.png")

        st.markdown('''
    # **FREE CONSULTATION**
    If you like what you see dont hesitate to contact us for a chat ⬇︎!
    ''')
        st.button("FREE CONSULTATION")

    
        






        
