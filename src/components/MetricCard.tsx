type Props = {
  label: string;
  value: string;
  hint?: string;
};

function MetricCard({ label, value, hint }: Props) {
  return (
    <article className="metric-card">
      <p className="eyebrow">{label}</p>
      <h3>{value}</h3>
      {hint && <p className="muted">{hint}</p>}
    </article>
  );
}

export default MetricCard;
