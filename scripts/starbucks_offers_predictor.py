import pandas as pd
import sys
import ast
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def clean_string_list(text):
    return str(text).replace('[', '').replace(']', '').replace(' ', '')


def coef_weights(lm_model, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)

    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model
    coefs_df['abs_coefs'] = np.abs(lm_model)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df


def transform_portfolio(df_portfolio):
    """
    The function makes the needed hot encodings for the portfolio dataset
    """
    # Enconde the column channels
    df_portfolio['channels'] = df_portfolio['channels'].apply(lambda row: clean_string_list(row))
    dummy_channels = df_portfolio['channels'].str.get_dummies(sep=",")
    df_portfolio = pd.concat([df_portfolio, dummy_channels], axis=1)
    # Encode the column offer_type
    df_portfolio['type_offer'] = df_portfolio['offer_type']
    df_portfolio_encoded = pd.get_dummies(df_portfolio, columns=['offer_type'])
    df_portfolio_encoded.drop(['channels'], axis=1, inplace=True)
    return df_portfolio_encoded


def transform_transcript(df_transcript):
    """
    The function transform the transcript dataframe with hot-encodings and returns separated dataframes for all
    transcripts, the sum of transactions grouped by user and the offers.
    """

    # Create a new column with time in days
    df_transcript['time_days'] = df_transcript['time']/24

    # Replace some tricky blanc spaces
    df_transcript['value'] = df_transcript['value'].apply(lambda x: str(x).replace('offer id', 'offer_id'))
    # Convert all the values in value column in dictionaries
    df_transcript['value'] = df_transcript['value'].apply(lambda x: ast.literal_eval(x))

    df_transcript['offer_received'] = df_transcript['event'].apply(lambda x: 1 if x == 'offer received' else 0)
    df_transcript['offer_completed'] = df_transcript['event'].apply(lambda x: 1 if x == 'offer completed' else 0)
    df_transcript['offer_viewed'] = df_transcript['event'].apply(lambda x: 1 if x == 'offer viewed' else 0)

    df_transactions = df_transcript.loc[df_transcript['event'].isin(['transaction'])].copy().reset_index(drop=True)
    df_offers = df_transcript.loc[df_transcript['event'].isin(['offer completed','offer received','offer viewed'])].copy().reset_index(drop=True)

    df_transactions['amount'] = df_transactions['value'].apply(lambda x: float(x['amount']))
    df_offers['offer_id'] = df_offers['value'].apply(lambda x: x['offer_id'])

    # Transactions is a one many relation. One user could have several transactions, then we aggregate them
    df_transactions_grouped = df_transactions.groupby(['person']).agg({'time': sum,
                                                                       'time_days': sum,
                                                                       'amount': sum}).sort_values(by='amount', ascending=False).reset_index(level=0)
    df_offers.drop(['value'], axis=1, inplace=True)
    return df_transcript, df_transactions_grouped, df_offers


def transform_profile(df_profile):
    """
    :param df_profile:
    :return: profile dataset transformed
    """

    df_profile = df_profile[df_profile['age'] < 81]
    df_profile = df_profile.drop(df_profile[df_profile['age'].isnull()].index)
    df_profile = df_profile.drop(df_profile[df_profile['gender'].isnull()].index)

    df_profile = df_profile.loc[df_profile['gender'].isin(['M','F'])].reset_index(drop=True)
    df_profile['gender_map'] = df_profile['gender'].apply(lambda x: 1 if x == 'M' else 0)

    df_profile['became_member_on'] = df_profile['became_member_on'].astype('str')
    df_profile['became_member_on'] = df_profile['became_member_on'].astype('datetime64[ns]')
    df_profile['start_year'] = df_profile['became_member_on'].dt.year
    df_profile["start_month"] = df_profile['became_member_on'].dt.month

    return df_profile


def combining_datasets(df_offers, df_profile, df_portfolio, df_transactions):
    """
    :param df_offers:
    :param df_profile:
    :param df_portfolio:
    :param df_transactions:
    :return: all the data merged
    """

    # This will be an innerjoin based on user id and users only users who match will be added (missing users deleted)
    offers_profile = pd.merge(df_offers, df_profile, left_on='person', right_on='id', how='inner')

    # Now the result is merged on portfolio via innerjoin (Keep only existent portfolio values)
    adding_portfolio = pd.merge(offers_profile, df_portfolio, left_on='offer_id', right_on='id', how='inner')

    adding_transactions = pd.merge(adding_portfolio, df_transactions, on='person', how='left')

    return adding_transactions


def load_data(portfolio_path, profile_path, transcript_path):
    """
    It loads the datasets
    :param profile_path
    :param portfolio_path
    :param transcript_path
    :return the dataframes content

    """
    portfolio = pd.read_json(portfolio_path, orient='records', lines=True)
    profile = pd.read_json(profile_path, orient='records', lines=True)
    transcript = pd.read_json(transcript_path, orient='records', lines=True)

    return portfolio, profile, transcript


def clean_data(portfolio, profile, transcript):
    """
    It receives raw datasets and process them

    :param portfolio
    :param profile
    :param transcript
    :return: X and y vectors needed to fit algorithms models
    """

    df_portfolio_encoded = transform_portfolio(portfolio)
    df_transcript_mix_encoded, df_transactions, df_offers = transform_transcript(transcript)
    df_profile_encoded = transform_profile(profile)

    all_data = combining_datasets(df_offers, df_profile_encoded, df_portfolio_encoded, df_transactions)
    matrix_plot = all_data.dropna(subset=['amount'])

    # Filter the only features needed for the algorithms training

    y = matrix_plot['offer_completed']
    X = matrix_plot.filter(['offer_received',
                            'offer_viewed',
                            'age',
                            'income',
                            'gender_map',
                            'start_month',
                            'reward',
                            'difficulty',
                            'duration',
                            'email',
                            'mobile',
                            'social',
                            'web',
                            'offer_type_bogo',
                            'offer_type_discount',
                            'offer_type_informational',
                            'amount'])

    return X, y


def main():
    if len(sys.argv) == 4:

        portfolio_path, profile_path, transcript_path = sys.argv[1:]

        print('Loading data...\n    PORTFOLIO: {}\n    PROFILE: {}\n TRANSCRIPT: {}'
              .format(portfolio_path, profile_path, transcript_path))
        portfolio, profile, transcript = load_data(portfolio_path, profile_path, transcript_path)

        print('Cleaning data...')
        X, y = clean_data(portfolio, profile, transcript)

        print('Training data...\n')

        # Create the test and train sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        coefficients = lm_model.coef_

        #Predict using your model
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X.shape, y.index.shape)

        print('Predicting data results...\n')

        #Score using your model
        test_score = r2_score(y_test, y_test_preds)
        train_score = r2_score(y_train, y_train_preds)

        #traing score
        print('Train Score:', float(train_score))
        #test score
        print('Test Score', float(test_score), '\n')

        coef_df = coef_weights(coefficients, X_train)

        print(coef_df.head(20).to_string())

        # Save the results
        coef_df.to_excel('Relevance_Features_Results.xlsx')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python clean_data.py ' \
              'data/portfolio.json data/profile.json ' \
              'data/transcript.json')


if __name__ == '__main__':
    main()
