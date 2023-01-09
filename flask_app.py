import pandas as pd
from flask import Flask, request, jsonify
# from functions.feature_extraction import get_features
from functions.scoring_model import read_data, load_model, get_coefficients
from functions.feature_extraction import select_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

import json

data = read_data(path="Data/data500.csv")

# data = read_data()
# features = get_features(data)
features = ['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_EMP_PHONE',
            'FLAG_WORK_PHONE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
            'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_12', 'AMT_REQ_CREDIT_BUREAU_YEAR',
            'NAME_CONTRACT_TYPE_Cash loans', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Working',
            'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_FAMILY_STATUS_Civil marriage',
            'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Unknown', 'NAME_HOUSING_TYPE_With parents',
            'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_Low-skill Laborers',
            'ORGANIZATION_TYPE_Business Entity Type 3', 'ORGANIZATION_TYPE_Self-employed', 'ANNUITY_INCOME_PERC',
            'PAYMENT_RATE', 'BURO_DAYS_CREDIT_MIN', 'BURO_DAYS_CREDIT_MAX', 'BURO_DAYS_CREDIT_MEAN',
            'BURO_DAYS_CREDIT_ENDDATE_MIN', 'BURO_DAYS_CREDIT_ENDDATE_MAX', 'BURO_DAYS_CREDIT_ENDDATE_MEAN',
            'BURO_DAYS_CREDIT_UPDATE_MEAN', 'BURO_AMT_CREDIT_SUM_OVERDUE_MEAN', 'BURO_CREDIT_ACTIVE_Active_MEAN',
            'BURO_CREDIT_ACTIVE_Bad debt_MEAN', 'BURO_CREDIT_ACTIVE_nan_MEAN', 'BURO_CREDIT_CURRENCY_currency 4_MEAN',
            'BURO_CREDIT_CURRENCY_nan_MEAN', 'BURO_CREDIT_TYPE_Credit card_MEAN',
            'BURO_CREDIT_TYPE_Interbank credit_MEAN',
            'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN', 'BURO_CREDIT_TYPE_Microloan_MEAN',
            'BURO_CREDIT_TYPE_Mobile operator loan_MEAN', 'BURO_CREDIT_TYPE_nan_MEAN', 'ACTIVE_DAYS_CREDIT_MIN',
            'ACTIVE_DAYS_CREDIT_MAX', 'ACTIVE_DAYS_CREDIT_MEAN', 'ACTIVE_DAYS_CREDIT_UPDATE_MEAN',
            'ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN', 'CLOSED_DAYS_CREDIT_MIN', 'CLOSED_DAYS_CREDIT_MAX',
            'CLOSED_DAYS_CREDIT_MEAN', 'CLOSED_DAYS_CREDIT_ENDDATE_MIN', 'CLOSED_DAYS_CREDIT_UPDATE_MEAN',
            'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN', 'PREV_DAYS_DECISION_MIN', 'PREV_DAYS_DECISION_MEAN',
            'PREV_CNT_PAYMENT_MEAN', 'PREV_CNT_PAYMENT_SUM', 'PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN',
            'PREV_NAME_CONTRACT_TYPE_nan_MEAN', 'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN',
            'PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
            'PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN',
            'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN', 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
            'PREV_NAME_CONTRACT_STATUS_nan_MEAN', 'PREV_NAME_PAYMENT_TYPE_XNA_MEAN', 'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
            'PREV_CODE_REJECT_REASON_HC_MEAN', 'PREV_CODE_REJECT_REASON_LIMIT_MEAN', 'PREV_CODE_REJECT_REASON_SCO_MEAN',
            'PREV_CODE_REJECT_REASON_SCOFR_MEAN', 'PREV_CODE_REJECT_REASON_nan_MEAN', 'PREV_NAME_TYPE_SUITE_nan_MEAN',
            'PREV_NAME_CLIENT_TYPE_New_MEAN', 'PREV_NAME_CLIENT_TYPE_nan_MEAN', 'PREV_NAME_GOODS_CATEGORY_Animals_MEAN',
            'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN', 'PREV_NAME_GOODS_CATEGORY_XNA_MEAN',
            'PREV_NAME_GOODS_CATEGORY_nan_MEAN', 'PREV_NAME_PORTFOLIO_Cards_MEAN', 'PREV_NAME_PORTFOLIO_XNA_MEAN',
            'PREV_NAME_PORTFOLIO_nan_MEAN', 'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN', 'PREV_NAME_PRODUCT_TYPE_nan_MEAN',
            'PREV_CHANNEL_TYPE_AP+ (Cash loan)_MEAN', 'PREV_CHANNEL_TYPE_nan_MEAN',
            'PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN', 'PREV_NAME_SELLER_INDUSTRY_XNA_MEAN',
            'PREV_NAME_SELLER_INDUSTRY_nan_MEAN', 'PREV_NAME_YIELD_GROUP_XNA_MEAN', 'PREV_NAME_YIELD_GROUP_high_MEAN',
            'PREV_NAME_YIELD_GROUP_nan_MEAN', 'PREV_PRODUCT_COMBINATION_Card Street_MEAN',
            'PREV_PRODUCT_COMBINATION_Cash_MEAN', 'PREV_PRODUCT_COMBINATION_Cash Street: high_MEAN',
            'PREV_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN', 'APPROVED_DAYS_DECISION_MIN',
            'APPROVED_DAYS_DECISION_MEAN', 'POS_MONTHS_BALANCE_MEAN', 'POS_SK_DPD_MAX', 'POS_SK_DPD_MEAN',
            'POS_NAME_CONTRACT_STATUS_XNA_MEAN', 'POS_NAME_CONTRACT_STATUS_nan_MEAN', 'INSTAL_DPD_MEAN',
            'INSTAL_PAYMENT_PERC_MAX', 'INSTAL_PAYMENT_DIFF_MAX',
            'INSTAL_PAYMENT_DIFF_MEAN', 'INSTAL_PAYMENT_DIFF_SUM', 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
            'INSTAL_DAYS_ENTRY_PAYMENT_SUM']
# print("______________________", len(features))
data = data[features + ["SK_ID_CURR", "TARGET"]].copy()
data = data.dropna()
X = data[features].values
X_sc = StandardScaler().fit_transform(X)
# print("_____________________", X_sc.shape)
model = load_model()
# print("_____________________")

coefficients = get_coefficients()
# print(coefficients)

# instantiate Flask object
flask_app = Flask(__name__)


@flask_app.route("/")
def index():
    return "model and data loaded ..."


#  local address: http://127.0.0.1:5000/client_score/? =116905
@flask_app.route('/client_score/')
def get_score():
    client_id = int(request.args.get('SK_ID_CURR'))
    # print('client_id =' , client_id)
    client_data = data[data['SK_ID_CURR'] == client_id]
    client_idx = client_data.index
    if client_data.shape[0] == 0:
        score = pd.DataFrame(data=[-1], columns=["score"])
    else:
        score = pd.DataFrame(data=[int(model.predict(X_sc[client_idx])[0])], columns=["score"])
    print(score)
    return jsonify(json.loads(score.to_json()))


# list of clients ids  (json file)
@flask_app.route('/clients_ids/')
def clients_ids_list():
    clients_id_list = data["SK_ID_CURR"].sort_values()
    # Convert to JSON
    clients_id_list_json = json.loads(clients_id_list.to_json())
    #
    # jsonify is a Flask method to return JSON data properly .
    return jsonify(clients_id_list_json)


# local address : http://127.0.0.1:5000/client_id_data/?SK_ID_CURR=116905
@flask_app.route('/client_id_data/')
def get_id_data():
    selected_id_client = int(request.args.get('SK_ID_CURR'))
    features = data[data['SK_ID_CURR'] == selected_id_client].drop(columns=["TARGET"]).loc[:, :]
    status = data.loc[data['SK_ID_CURR'] == selected_id_client, "TARGET"]
    # Convert  to JSON
    features_json = json.loads(features.to_json())
    status_json = json.loads(status.to_json())
    # Returning the processed data
    return jsonify({'status': status_json, 'data': features_json}) \
 \
        # @flask_app.route('/features/')

#@flask_app.route('/features/')
 #def get_features():
 #  n = int(request.args.get('n'))
 # c_df = list(model.named_steps["logistic"].coef_[0])
  #c_df_abs = [abs(c) for c in c_df]
  #c_df_abs = sorted(c_df_abs, reverse=True)[:n]
  #c_df = [c for c in c_df if abs(c) in c_df_abs]
#model_importance = pd.DataFrame(zip(features, c_df), columns=["feature", "importance"])
# model_importance = model_importance.sort_values("importance", ascending=False)
# return jsonify(json.loads(model_importance.to_json()))


@flask_app.route('/global_importance/')
def plot_g_importance():
    n = int(request.args.get('n'))
    c_df = list(model.named_steps["logistic"].coef_[0])
    c_df_abs = [abs(c) for c in c_df]
    c_df_abs = sorted(c_df_abs, reverse=True)[:n]
    c_df = [c for c in c_df if abs(c) in c_df_abs]
    model_importance = pd.DataFrame(zip(features, c_df), columns=["feature", "importance"])
    model_importance = model_importance.sort_values("importance", ascending=False)
    return jsonify(json.loads(model_importance.to_json()))

    # n_features = request.args.get('n')
    # print('n_features =', n_features)
    # coef = list(model.named_steps["logistic"].coef_[0])
    # coef_abs = [abs(c) for c in coef]
    # coef_abs = sorted(coef_abs, reverse=True)[:n_features]
    # coef = [c for c in coef if abs(c) in coef_abs]
    # model_importance = pd.DataFrame(zip(features, coef), columns=["feature", "importance"])
    # model_importance = model_importance.sort_values("importance", ascending=False)
    # model_importance = model_importance.to_json()
    # return jsonify({'model_importance': model_importance})
    # return jsonify(n_features.to_json())


def get_Xtrain():
    """
    :return:
    """
    df = data
    y = df["TARGET"].values
    X_t = train_test_split(X_sc, y, test_size=0.3)[0]
    return X_t, X_sc


@flask_app.route('/local_importance/')
def plot_l_importance():
    client_id = int(request.args.get('SK_ID_CURR'))
    # print('client_id =', client_id)
    client_data = data[data['SK_ID_CURR'] == client_id]
    # num_features = int(request.args.get('n'))
    y = data["TARGET"].values
    X_ = data[features].values
    X_sc_ = StandardScaler().fit_transform(X_)
    X_t = train_test_split(X_sc_, y, test_size=0.3)[0]

    #print('*****************')
    #print(data[data['SK_ID_CURR'] == client_id])
    if data[data['SK_ID_CURR'] == client_id].empty:
        return 'not available'
    else:
        # X_train, X_sc_ = get_Xtrain(data, features)
        explainer = LimeTabularExplainer(X_t, mode="classification", class_names=['Accepted', 'Refused'], feature_names= features)
        idx = data[data["SK_ID_CURR"] == client_id].index
        data_instance = X_sc[idx].reshape(len(features), )
        explanation = explainer.explain_instance(data_instance, model.predict_proba, num_features=10)
        sel_features = select_features(explanation)
        importance = explanation.as_list()
        df = pd.DataFrame(importance,columns=['features','importance'])
        print(df)
        # sel_features = features
    return jsonify(json.loads(df.to_json()))
    #return jsonify(sel_features, importance)

    #############


if __name__ == "__main__":
    flask_app.run()
