export PYTHONPATH=.

# run construction benchmarks
for i in {2..10}; do
  if [ $i -lt 10 ]; then
    track="track_0${i}.t"
  else
    track="track_${i}.t"
  fi
  python -m src.benchmark --target construction --track ${track} --depth 3 --fast
done

# run bfs benchmarks
for i in {2..10}; do
  if [ $i -lt 10 ]; then
    track="track_0${i}.t"
  else
    track="track_${i}.t"
  fi
  python -m src.benchmark --target bfs --track ${track} --fast
done

