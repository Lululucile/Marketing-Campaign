# Part 1 Introduction
The targeting of client groups with personalized promotional activities is an essential part of customer relationship management (Reutterer et al., 1987). Inherently, individuals have distinct value consideration sets and respond differently to incentives (Abaluck and Adams-Prassl, 2021). To optimize return on marketing spending, marketers should consider Direct marketing directing certain promotional campaigns to specific segments of consumers who are most "reactive" to the offer, to subsequently induce sales and profit advancements (Wong et al., 2005). Else, excessive marketing resources may be wasted on inattentive customers (those who do not value the specific promotion campaign) or profit margins may be eroded by offering an overly generous incentive (a lower discount may suffice for certain groups for inducing sales) (Wong et al., 2005). In this study, we approached the task of understanding the key demographic composition of an existing marketing campaign’s response rate by 
1. Using RFM (Recency - Frequency - Monetary Value) for segmentation; 
2. Formulating a response model to forecast who will respond to an offer for a product or service. 
For future targeted marketing campaigns, this pilot study's data may be utilized to develop machine learning algorithms that accurately interpret and predict a member's likelihood of responding. In essence, the business problem this research focuses on can be stated as - __Maximizing response rate of marketing campaign by direct targeting promotion-sensitive customers__.

# Part 2 Data Preparation
The Marketing Campaign Dataset, available on Kaggle, consisted of detailed demographic information, and whether an individual customer responded when a particular (undisclosed) marketing campaign was presented across 29 columns and 2240 rows. (Exhibit 1) contains detailed elaborations of each column key.

__Data Cleaning__: 24 out of 2240 missing value in observations were identified and removed; ID variable were dropped since it is useless in subsequent analysis; Categorical variables “education” and “Marital_Status” are changed into factors; “Dt_Customer” changed to date format; Variables “Z_Cost” and “ Z_Revenue” with all the same values are dropped. To make sure the accuracy of the prediction model, we also identify and drop outliers in income and Dt_customers.

__Formulation of new variables: 
1. **Age** = 2014 - “Year_Birth” ; Assumption: the data was collected in 2014
2. **Memdate (Years of being member)** = 2014 - Year (Dt.Customer) 
3. **RFM segmentation** is a scoring technique used to better quantify customer behavior (Karaman, 2019), to conduct RFM analysis following variables are created: 
● Recency = Number of days since the last purchase. 
● Frequency = Total transaction value added together, which includes number of deal purchases, number of web purchases, number of catalog purchases, number of store purchases. 
● Monetary = Total dollar amount added for all the products being bought by each
customer, which include Wine, Fruits, Meat, Fish, Sweets, Gold Products. 
4. **Customer Life Time Value (CLV)** is the economic value created by a company's
relationship with its customers, concerning the present value of future cash flows related to the customer relationship. In other words, CLV calculates the net profit that a customer will generate for the organization over time. As a consequence, historical customer transactions may be used to forecast the monetary value of a company's client relationship.
**CLV = $M[ r/(1+d-r)]**
_Modeling and detailed calculation of the RFM and CLV variables are elaborated in their respective sections (part 4) below_.

# Part 3 Exploratory data analysis
## EDA based on customers demographic:
Base on the data given, some interesting demographic characteristics were found from the data. The sample appears to have an mean age group of 45 with a standard deviation of 12.(Exhibit2&3). So middle age people are our largest target customers. Looking into the education of the group, the customers have a high education level background with college graduates being the largest group and following by PhD.(Exhibit4). The customers are mostly married and together(Exhibit5). The income are ranging from $1730 to $98,777, with a average of income are $51500.(Exhibit6). Most of the family have 0 or 1 young children, and very few with 2 children.(Exhibit7). Similar situation with teenage in household, most with 1 or 2 teenage and very few with 2.(Exhibit8). A correlation graph was plotted(Exhibit9), and focused more on the “response” column. There is higher correlation with the amount spent in wine and in meat, and number of purchase made using catalogue, number of purchase through websites, and also if they accepted offers in previous campaign, among previous campaign, people who accepted in campaign 5 and 1 have a higher correlation.

CLV was calculated using formula CLV = $M[ r/(1+d-r) where we assume the gross margin is 40%, and with a discount rate of 10%. It is proved that not all customers are equally profitable. It was found that 28.5% of the customers are consuming 80% percent of the total CLV. The other 63% consumed 19.5% of the CLV, and the remainder of a firm’s customers are almost unprofitable. So instead of spending money on the unproditbal customer group, those 28.5% of the profit group should be our target group and doing campaigns.

## RFM: Customer segmentation
Exhibit 10 shows process chart of how RFM analysis is being conducted for this case. Recency measures the freshness of the customer activity. Frequency measures how frequent customer is making a purchase. Lastly, monetary measures the purchase power of a customer After each customer’s Recency, Frequency, Monetary value is computed, an independent sorting method with quantiles, is used to segment the customers. 

As shown in exhibit 11, if a customer has a recency value fell between the min and the first quantile, which is 0-24, the customer would have a R score of 1. Same applies for frequency and Monetary. 

Theoretically, there should be 4*4*4=64 segments generated. However, since we do not have a large enough sample size, meaning that not every segments are filled with customers, there are only 38 segments being generated. The output for the segmentations are shown in the exhibit 12.

The segment which has highest response rate are 4/2/4. This group of customers has high recency, and median frequency, and high monetary scores. When examining the patterns of common behaviors for segments having a higher response rate, a discovery is found that the higher the monetary score they have, the more likely they are going to respond to a campaign. To identify which of the resultant RFM cells to target, a breakeven analysis is used to identify the cut-off response rate.

**Breakeven purchase rate = cost to offer the campaign/average profit from the single sale** Breakeven purchase rate being computed should be the threshold to determine which segments of customers to target. Customers who have a higher response rate than breakeven purchase rate should be seen as a firm's value customers. Since not all the customers are equally valuable to the firm, RFM is critical to conduct to generate a better understanding of customer behavior and to identify who are the most valuable customers.

# Part 4 Model estimation and results
## Prediction Modeling
Based on Supervised Data, we use Logistic Regression and Classification Tree to build prediction models. 

Modeling iterations are in Exhibit 12. The following logistic model was selected as the best suited model since it had higher AUC and accuracy. Here, monetary and frequency are into consideration along with demographics. Evaluation below-

![Pictures in Readme/Part 4.png](https://github.com/Lululucile/Marketing-Campaign/blob/main/Pictures%20in%20Readme/Part%204.png)

log odds(Response) = β0 + β1Year_Birth + β2EducationBasic + β3EducationGraduation + β4EducationMaster + β5EducationPhD + β6Marital_StatusAlone + β7Marital_StatusDivorce + β8Marital_StatusMarried + β9Marital_StatusSingle + β10Marital_StatusTogether + β11Marital_StatusWidow + β12Marital_StatusYOLO + β13Income + β14Kidhome + β15Teenhome + β16Dt_Customer + β17Recency + β18AcceptedCmp3 + β19AcceptedCmp4 + β20AcceptedCmp5 + β21AcceptedCmp1 + β22AcceptedCmp2 + β23Complain + β24Monetary + β25Frequency

# Part 5 - Results and managerial takeaways
## Model deployment
Retailer has a mid-age and highly education customers. However, only few customers (~29%) contribute to the revenue, and it was seen that customers with high monetary values are more likely to respond to marketing campaigns.

Hence, retailer should not target everyone, and can use the prediction model to only target people who are likely to respond to the marketing campaign such that retailer reduces it’s cost and maximizes revenue. This will help the retailer to allocate market campaigns more efficiently and effectively.

# Part 6 Limitations
Before deploying the model, retailer should be aware that there will be an underlying bias for those customers who are regular customers and would have gone to the retailer even without the promotion. Previous transaction history is needed to understand how much “additional” revenue was gained from the marketing campaign too gauge success.

![Pictures in Readme/Part 6.png](https://github.com/Lululucile/Marketing-Campaign/blob/main/Pictures%20in%20Readme/Part%206.png)

Also, from the above graph, as the threshold is changed, there is a tradeoff between True Positive Rate and False Positive Rate. If it is discovered that cost of targeting is greater than the missed opportunity (i.e., not targeting a potential customer), then retailer should increase the threshold to greater than 0.5, so to decrease the False positive. Since there is not enough information on the costs, adjusting the threshold (currently at 0.5) to account for the tradeoff between false negatives and false positives cannot be addressed at this moment.
