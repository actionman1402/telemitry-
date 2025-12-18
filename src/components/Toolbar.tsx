type Props = {
  value: string;
  onChange: (value: string) => void;
};

const filters = [
  { label: 'All events', value: 'all' },
  { label: 'Start', value: 'start' },
  { label: 'Record', value: 'record' }
];

function Toolbar({ value, onChange }: Props) {
  return (
    <div className="toolbar">
      <span className="muted">Event filter</span>
      <div className="pill-group">
        {filters.map((filter) => (
          <button
            key={filter.value}
            className={value === filter.value ? 'pill active' : 'pill'}
            onClick={() => onChange(filter.value)}
          >
            {filter.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default Toolbar;
