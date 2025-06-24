FILE=$1

if [ -f "$FILE" ]; then
    echo "$FILE found."
else 
    echo "$FILE does not exist."
fi

sed -i '/SPACE/d' $FILE
