from sklearn.metrics import accuracy_score, f1_score

def test(classifier, X_train, y_train, X_test, y_test, name, verbose=True):
    classifier.fit(X_train, y_train)
    
    y_pred_train = classifier.predict(X_train)
    train_scores = [
              accuracy_score(y_train, y_pred_train), 
              f1_score(y_train, y_pred_train, average=None),
             ]
    
    y_pred_test = classifier.predict(X_test)
    val_scores = [ 
              accuracy_score(y_test, y_pred_test), 
              f1_score(y_test, y_pred_test, average=None),
             ]
    
    if verbose:
        print("TRAIN")
        print(f"r^2: {train_scores[0]}")
        print(f"accuracy: {train_scores[1]}")
        print(f"f1: {train_scores[2]}")
        print()
        print("VALIDATION")
        print(f"r^2: {val_scores[0]}")
        print(f"accuracy: {val_scores[1]}")
        print(f"f1: {val_scores[2]}")

    return y_pred_train, train_scores, y_pred_test, val_scores