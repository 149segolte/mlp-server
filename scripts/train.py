import sys
import os
import json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve


if __name__ == '__main__':
    input_file = sys.argv[1]
    f = open(input_file, 'r')
    data = json.load(f)
    f.close()
    '''
        structure of data is:
        z.object({
            name: z.string().min(1),
            id: z.string().uuid(),
            project: z.string().uuid(),
            file: z.string(),
            target: z.string(),
            multi_class: z.boolean(),
            model: z.enum([
                'Ordinary Least Squares',
                'Elastic Net',
                'Logistic Regression',
                'Gradient Descent',
                'Decision Tree',
                'Support Vector Machine'
            ]),
            test_size: z.coerce.number().min(0).max(1),
            alpha: z.coerce.number().min(0).optional(),
            l1_ratio: z.coerce.number().min(0).max(1).optional(),
            max_iter: z.coerce.number().min(1).optional(),
            tol: z.coerce.number().min(0).optional(),
            C: z.coerce.number().min(0).optional(),
            loss: z.optional(
                z.enum([
                    'hinge',
                    'huber',
                    'log_loss',
                    'squared_hinge',
                    'perceptron',
                    'epsilon_insensitive',
                    'squared_epsilon_insensitive'
                ])
            ),
            shuffle: z.boolean().optional(),
            learning_rate: z.enum(['constant', 'optimal', 'invscaling', 'adaptive']).optional(),
            early_stopping: z.boolean().optional(),
            validation_fraction: z.coerce.number().min(0).max(1).optional(),
            criterion: z.enum(['gini', 'entropy', 'log_loss']).optional(),
            max_depth: z.coerce.number().min(0).optional(),
            min_samples_split: z.coerce.number().min(0).optional(),
            min_samples_leaf: z.coerce.number().min(0).optional(),
            max_features: z.enum(['auto', 'sqrt', 'log2']).optional(),
            max_leaf_nodes: z.coerce.number().min(0).optional(),
            kernel: z.enum(['linear', 'poly', 'rbf', 'sigmoid']).optional(),
            degree: z.coerce.number().min(0).optional(),
            gamma: z.enum(['scale', 'auto']).optional()
        })
    '''

    # Load the data
    df = pd.read_csv(data['file'])
    print(df.info())

    # Split the data into training and testing sets
    X = df.drop(data['target'], axis=1)
    y = df[data['target']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data['test_size'])

    # Train the model
    if data['model'] == 'Ordinary Least Squares':
        model = LinearRegression()
    elif data['model'] == 'Elastic Net':
        model = ElasticNet(
            alpha=data['alpha'],
            l1_ratio=data['l1_ratio'],
            max_iter=data['max_iter'],
            tol=data['tol'])
    elif data['model'] == 'Logistic Regression':
        if data['multi_class']:
            model = LogisticRegression(
                max_iter=data['max_iter'],
                tol=data['tol'],
                C=data['C'],
                multi_class='multinomial')
        else:
            model = LogisticRegression(
                max_iter=data['max_iter'],
                tol=data['tol'],
                C=data['C'])
    elif data['model'] == 'Gradient Descent':
        model = SGDClassifier(
            loss=data['loss'],
            shuffle=data['shuffle'],
            alpha=data['alpha'],
            eta0=0.1,
            max_iter=data['max_iter'],
            tol=data['tol'],
            learning_rate=data['learning_rate'],
            early_stopping=data['early_stopping'],
            validation_fraction=data['validation_fraction'])
    elif data['model'] == 'Decision Tree':
        model = DecisionTreeClassifier(
            criterion=data['criterion'],
            max_depth=data['max_depth'],
            min_samples_split=data['min_samples_split'],
            min_samples_leaf=data['min_samples_leaf'],
            max_features=data['max_features'],
            max_leaf_nodes=data['max_leaf_nodes'])
    elif data['model'] == 'Support Vector Machine':
        model = SVC(
            C=data['C'],
            kernel=data['kernel'],
            degree=data['degree'],
            gamma=data['gamma'])

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print('Score:', score)

    # Get metrics
    attrs = {}
    if data['model'] == 'Ordinary Least Squares':
        attrs['intercept'] = model.intercept_.tolist()
        attrs['coefficients'] = model.coef_.tolist()
    elif data['model'] == 'Elastic Net':
        attrs['intercept'] = model.intercept_.tolist()
        attrs['coefficients'] = model.coef_.tolist()
    elif data['model'] == 'Logistic Regression':
        attrs['intercept'] = model.intercept_.tolist()
        attrs['coefficients'] = model.coef_.tolist()
        attrs['classes'] = model.classes_.tolist()
    elif data['model'] == 'Gradient Descent':
        attrs['intercept'] = model.intercept_.tolist()
        attrs['coefficients'] = model.coef_.tolist()
        attrs['classes'] = model.classes_.tolist()
    elif data['model'] == 'Decision Tree':
        attrs['feature_importances'] = model.feature_importances_.tolist()
        attrs['classes'] = model.classes_.tolist()
    elif data['model'] == 'Support Vector Machine':
        attrs['classes'] = model.classes_.tolist()
        attrs['dual_coef'] = model.dual_coef_.tolist()
        attrs['intercept'] = model.intercept_.tolist()
        attrs['support_vectors'] = model.support_vectors_.tolist()
        attrs['support_'] = model.support_.tolist()
        attrs['n_support'] = model.n_support_.tolist()

    metrics = {
        'score': score,
        'attrs': attrs
    }

    # Get performance metrics
    perf = {}
    if data['model'] == 'Ordinary Least Squares' or data['model'] == 'Elastic Net':
        perf['explained_variance_score'] = explained_variance_score(
            y_test, model.predict(X_test))
        perf['max_error'] = max_error(y_test, model.predict(X_test))
        perf['mean_absolute_error'] = mean_absolute_error(
            y_test, model.predict(X_test))
        perf['mean_squared_error'] = mean_squared_error(
            y_test, model.predict(X_test))
        perf['median_absolute_error'] = median_absolute_error(
            y_test, model.predict(X_test))
        perf['r2_score'] = r2_score(y_test, model.predict(X_test))
    elif data['model'] == 'Logistic Regression' or data['model'] == 'Gradient Descent' or data['model'] == 'Decision Tree' or data['model'] == 'Support Vector Machine':
        perf['accuracy_score'] = accuracy_score(
            y_test, model.predict(X_test))
        perf['balanced_accuracy_score'] = balanced_accuracy_score(
            y_test, model.predict(X_test))
    metrics['perf'] = perf

    # Save the model to a file called id.pkl in folder ./.server/project_id/
    file_path = './.server/' + data['project'] + '/' + data['id'] + '.pkl'
    model_file = open(file_path, 'wb')
    model_file.write(pickle.dumps(model))

    # Save config
    print(df.head())
    f = open(input_file, 'w')
    config = {
        'id': data['id'],
        'name': data['name'],
        'model': data['model'],
        'project': data['project'],
        'file': os.path.abspath(file_path),
        'config': data,
        'metrics': metrics,
    }
    print(config)
    json.dump(config, f)
    f.close()
