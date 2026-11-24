top -H -b -n 1 -w 200 | awk 'NR>7 && $9>50 {print $1}' | while read t; do
  tg=$(awk '/^Tgid/{print $2}' /proc/$t/status 2>/dev/null)
  echo "tid=$t tgid=$tg thread_comm=$(cat /proc/$t/comm 2>/dev/null) proc=$(tr '\0' ' ' < /proc/$tg/cmdline 2>/dev/null | cut -c1-70) wchan=$(cat /proc/$t/wchan 2>/dev/null)"
done
echo "--- libs mapped by the hot tgids that match hsa/dxg:"
for tg in $(top -H -b -n 1 -w 200 | awk 'NR>7 && $9>50 {print $1}' | while read t; do awk '/^Tgid/{print $2}' /proc/$t/status; done | sort -u); do
  echo "tgid $tg: $(grep -oE '/[^ ]*(libhsa-runtime64|librocdxg|libamdhip64)[^ ]*' /proc/$tg/maps | sort -u | tr '\n' ' ')"
done
