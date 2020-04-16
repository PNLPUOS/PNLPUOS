# pnlp_study_project
Practical NLP for Survey Analysis with deepsight GmbH.
***

## Project Structure

<pre>
|-- data
    |-- raw
    |-- processed
|-- notebooks
    |-- sentiment_analysis
    |-- topic_clustering
|-- src
    |-- sentiment_classifier
</pre>
***

### Usage
1. Clone the repository.
2. Navigate to the repository directory.
3. Install dependencies. (pip install -r requirements.txt)
4. Obtain dataset available on slack. Place in 'data' directory.  
5. Obtain fasttext English model [here](https://fasttext.cc/docs/en/english-vectors.html). Place ‘common-crawl-300d-2M-subword’ in 'src'.		
6. Navigate to src and test run the main script. (python main.py)
***

### Experiment Logging
To access the full capabilities of the script you should integrate the sacred experiment with the MongoDB Atlas cluster prepared for this project. Sacred is an experiment logging toolkit which easily integrates into scripts and records configurations, results, files, etc. You can find more information on Sacred [here](https://sacred.readthedocs.io/en/stable/).

First, provide your email to the MongoDB Atlas cluster owner. You will receive an invite to register and gain access to the project.  

Next, adjust the credentials.py file in the src directory by adding the username and password you used to register for MongoDB Atlas. You should now be able to test that your experiment logs to the database by uncommenting the experiment observer in main.py. If the script runs without problems you have successfully logged the experiment.
***

### Viewing Experiment Results
To view experiment results logged to the database you must install Omniboard. Omniboard is a browser based application optimized for viewing experiments logged via sacred and MongoDB.

Omniboard requires node.js. First, install node.js [here](https://nodejs.org/en/download/ ).

Next, use the terminal to install Omniboard. (npm install -g omniboard).
Run Omniboard (npx omniboard –mu url) You must replace 'url' with the database server url. The easiest way to obtain this is to run the main.py and copy-paste the output url into the terminal. If you receive no errors Omniboard should be running. 

To view the board and interact with the results, navigate to localhost:9000 in your browser. From Omniboard you can adjust the shown columns and other settings via the options menu at the top right of the dashboard.
![Image unavailable.](https://raw.githubusercontent.com/vivekratnavel/omniboard/master/docs/assets/screenshots/table.png)

Further information on setting up Omniboard can be found [here](https://vivekratnavel.github.io/omniboard/#/quick-start).
