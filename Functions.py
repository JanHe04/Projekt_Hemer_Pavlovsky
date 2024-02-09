def import_libraries():
    try:
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        import sklearn 
        import sklearn.datasets
        import statsmodels.api as sm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np
        import lightgbm
        print("All libraries are imported successfully.")

    except ImportError as e:
        library_name = str(e).split("No module named ")[-1].replace("'", "")
        print(f"Could not import the library {library_name}. Please ensure it is installed.")


def scrape_and_process_data(list_number1, list_number2, col_names, start_year, end_year):
    merged_df = pd.DataFrame(columns=col_names)

    for year in range(start_year, end_year):
        for ln1, ln2 in zip(list_number1, list_number2):
            response = requests.get(
                f"https://www.fortunaliga.cz/statistiky?unit=4&status=&parameter=1&season={year}&club=0&game_limit=&nationality=&age=&position=&list_number={ln1}&order=2&order_dir=1&list_number={ln2}"
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', class_='table')

                titles = [element.text for element in soup.find_all('abbr')]

                content = [td.text for tr in table.find_all('tr') for td in tr.find_all('td')]
                
                num_elements_per_row = 14
                num_rows = len(content) // num_elements_per_row
                rows = [content[i:i + num_elements_per_row] for i in range(0, len(content), num_elements_per_row)]
                df = pd.DataFrame(rows, columns=[f'Column_{i+1}' for i in range(num_elements_per_row)])
                df = df.set_index('Column_1')
                df.columns = titles
                df = df.reset_index()
                df = df.drop_duplicates(subset=['Column_1'])

                merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)

    extracted_columns = merged_df['Zápas'].str.extract(r'([A-Z]+)-([A-Z]+) (\d+):(\d+), (\d{2}/\d{2})')
    extracted_columns.columns = ['Home', 'Away', 'Home_Goals', 'Away_Goals', 'Date']
    final_df = pd.concat([extracted_columns, merged_df.drop('Zápas', axis=1)], axis=1)

    return final_df


def plot_total_goals_by_team(df):
    df['Total_Goals'] = df['Home_Goals'] + df['Away_Goals']

    home_goals = df.groupby('Home')['Home_Goals'].sum()
    away_goals = df.groupby('Away')['Away_Goals'].sum()

    teams = list(set(df['Home']).union(set(df['Away'])))
    teams_sorted = sorted(teams, key=lambda team: home_goals.get(team, 0) + away_goals.get(team, 0), reverse=True)

    plt.figure(figsize=(12, 6))

    for team in teams_sorted:
        home_goal = df[df['Home'] == team]['Home_Goals'].sum()
        away_goal = df[df['Away'] == team]['Away_Goals'].sum()

        plt.bar(team, home_goal, label=f'{team} - Home Goals', alpha=0.7, color='blue')
        plt.bar(team, away_goal, label=f'{team} - Away Goals', alpha=0.7, color='green', bottom=home_goal)

    plt.legend(['Home Goals', 'Away Goals'])
    plt.xlabel('Teams')
    plt.ylabel('Total Goals')
    plt.title('Total Goals Scored by Each Team (Sorted)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def significant_coefficients(coefficient_names, coefficients, p_values, significance_threshold=0.05):

    for name, coef, p_value in zip(coefficient_names, percentage_change, p_values):
        if p_value < significance_threshold:
            print(f'{name}: {coef:.2f}% (Statistically significant)')
        else:
            print(f'{name}: {coef:.2f}%')