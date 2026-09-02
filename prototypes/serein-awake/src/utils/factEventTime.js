export function factEventTimeLabel(item) {
  const startDate = item.local_date;
  const endDate = item.local_end_date || startDate;
  const startTime = item.local_start_time;
  const endTime = item.local_end_time || startTime;

  if (startDate !== endDate) {
    return `${startDate} ${startTime} → ${endDate} ${endTime}`;
  }
  return startTime === endTime
    ? `${startDate} · ${startTime}`
    : `${startDate} · ${startTime}–${endTime}`;
}

export function compareFactEventsByEnd(left, right) {
  const leftEnd = `${left.local_end_date || left.local_date}T${left.local_end_time || left.local_start_time}`;
  const rightEnd = `${right.local_end_date || right.local_date}T${right.local_end_time || right.local_start_time}`;
  return rightEnd.localeCompare(leftEnd) || String(right.item_id).localeCompare(String(left.item_id));
}
