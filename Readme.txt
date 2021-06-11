1. Set game-window in 1280x800 px resolution, place it in the top-right corner. I usually place camera on top of truck and
zooming-in navigator
2. Edit variable 'file_name' in 1-collect_data.py script to your preference
3. Run 1-collect_data.py script and play the game ('T' for pausing/resuming) it will collect screenshots and your
keys(W, S, A, D, WA, WD, SA, SD, No_keys) as npy arrays[500, [200, 300, 3], [9]]. len, img_size, keys
4. Run 2-balance_data.py script with edited directories to cut-off many W and NK choices to balance data a bit
5. Manually relocate some files from balanced data to a folder with validation data (if you need it)
6. Run 3-train_model_generator.py, set as many epochs you want(1 epoch = all balanced dataset available)
7. Set game-window in 1280x800 px resolution, place it in the top-right corner
8. Run 4-playmodel.py with path to trained model and switch back to game