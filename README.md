# OxML_ESG_BL_NC

Here is some information about the format of the module and the schedule:
Format:
We will be using Kaggle as a platform to host the challenge. You are encouraged to team up in group of 3 people. Working in small teams have several benefits:
you will likely be more engaged, therefore you will take more out of the program
You will learn faster by leveraging other’s experiences,
You will build up your network.
If you still want to work solo, it is possible.
Case selection: You will chose the case after the Case presentations. A group can only chose to work on a single ML case simultaneously.
Deliverable:
Before June 21st (end of the day), you will need to provide :
A submission file that includes model predictions
the code for building and training your model
a small presentation that explains your approach to solve the problem
There is a ranking based on model performance, but clarity on code and presentation greatly matter to be part of the award winning teams. The best projects will be showcased through the OxML platform.
Communication & Support:
Discussions between speakers and candidates will happen on Slack workspace
We highly value collaborative learning, and encourage groups to support each other through a dedicated "Support" Slack channel
Speakers will only answer most relevant/common questions that benefits every groups
Schedule :
[LIVE] 30/05 (13:00 UK time) - 1h : Case presentation
[LIVE] 31/05 (13:00 UK time) - 1h : Case presentation
31/05-02/06 - 1h : Case and group selection
02/06 - 21/06 : Work on the Case / Get support through Slack from other groups and speakers
21/06: Submit your Case for final evaluation
[LIVE] 30/06 (13:00 UK time) - 1h : Speakers share final ranking along with their feedback from they have seen from groups.



Objective: 
Shoot for high 80%s for accuracy on the classification. Once reached, move to objective 2 of table detector project. 

Prediction: 
1. One model for class and high accuracy of submission.csv 
2. ** Extra ** probabilty of strenth to the class 

Ideas: 
LDA topic modeling 

UMAP finding - 
The environment and social classes appear closely correlated and there may be confusion from prediction models on these two classes 

<img width="498" alt="Screenshot 2023-06-06 at 4 57 59 PM" src="https://github.com/Blodgic/OxML_ESG_BL_NC/assets/7229755/c845ee02-5dd2-44a0-99b7-b7b3956ebc0f">


Considerations: 
how to predict? 
- whole page
- paragraph 
- sentence 
- word

do we predict a probabilty with a classification for strenth to the class? 
Example: Class Prediciton: Governance Probabilty: 78% strength to class
