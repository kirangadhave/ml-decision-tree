echo "Question 1c:"
python3.6 $(dirname $0)/Code/run_id3.py Dataset/updated_train.txt Dataset/updated_train.txt
echo ""
echo ""
echo ""
echo ""

echo "Question 1d:"
python3.6 Code/run_id3.py Dataset/updated_train.txt Dataset/updated_test.txt
echo ""
echo ""
echo ""
echo ""

echo "Question 2:"
python3.6 Code/run_id3.py Dataset/updated_train.txt Dataset/updated_test.txt Dataset/Updated_CVSplits 1 2 3 4 5 8 10 15 20

