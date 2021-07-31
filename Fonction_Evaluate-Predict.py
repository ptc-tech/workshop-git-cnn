test_loss, test_acc = model.evaluate(images_test, labels_test, verbose=2)
predictions = model.predict(images_test)