#!/bin/sh

for d in ../output/*/; do 
  dname=$(echo "${d}" | awk -F/ '{ print $3}')
  echo $dname
  if [ -e "$d"test_results.tsv ]; then
    cp "$d"test_results.tsv ../sub/"$dname"_test_results.tsv
  fi
done

