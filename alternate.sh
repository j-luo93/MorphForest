cd py

i=0
for lang in eng tur ara ger; do
    top_affix=500;
    # for top_words in 2500 5000 10000 25000 50000 75000 100000; do
    for top_words in 10000; do
        for compounding in 0 1 ; do
            for correlation in 0 1; do
                echo $lang $top_affix $top_words $compounding 1;
                file=$lang-${top_affix}-$top_words-$compounding-1.results
                if [ -f alternate/$file ]; then
                    echo file exists;
                    continue;
                fi
                i=$((i+1));
                if [ $((i % 10)) -gt 0 ]; then
                    python run.py $lang $top_affix $top_words $compounding 1 $correlation 0.001 1.0 > alternate/$file &
                else
                    python run.py $lang $top_affix $top_words $compounding 1 $correlation 0.001 1.0 > alternate/$file ;
                fi
                sleep 30
            done
        done
    done
done
