import { useMemo, useState, type ReactNode } from 'react';
import { ArrowDown, ArrowUp } from 'lucide-react';
import { cx } from '../../lib/cx';

export interface Column<Row> {
  key: string;
  /** Header text, with units: "Mean time (s)". */
  header: string;
  /** Numbers are right-aligned in mono; text is left-aligned. */
  numeric?: boolean;
  render: (row: Row) => ReactNode;
  /** Provide to make the column sortable. */
  sortValue?: (row: Row) => number | string;
}

interface TableProps<Row> {
  columns: Column<Row>[];
  rows: Row[];
  rowKey: (row: Row) => string;
  caption: string;
  /** Source line shown under every table, e.g. "Source: README section 12". */
  source?: string;
  selectedKey?: string;
  onSelect?: (row: Row) => void;
  dense?: boolean;
}

type Sort = { key: string; dir: 'asc' | 'desc' } | null;

export function Table<Row>({ columns, rows, rowKey, caption, source, selectedKey, onSelect, dense = false }: TableProps<Row>) {
  const [sort, setSort] = useState<Sort>(null);

  const sorted = useMemo(() => {
    const col = sort && columns.find((c) => c.key === sort.key);
    if (!sort || !col?.sortValue) return rows;
    const get = col.sortValue;
    const factor = sort.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const va = get(a);
      const vb = get(b);
      return (va < vb ? -1 : va > vb ? 1 : 0) * factor;
    });
  }, [rows, columns, sort]);

  const toggleSort = (key: string) =>
    setSort((s) => (s?.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'asc' }));

  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto rounded border border-border">
        <table className="w-full border-collapse text-body">
          <caption className="sr-only">{caption}</caption>
          <thead className="bg-raised">
            <tr>
              {columns.map((col) => {
                const active = sort?.key === col.key;
                const ariaSort = active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined;
                return (
                  <th
                    key={col.key}
                    scope="col"
                    aria-sort={ariaSort}
                    className={cx('h-9 px-3 text-label text-text-2', col.numeric ? 'text-right' : 'text-left')}
                  >
                    {col.sortValue ? (
                      <button
                        type="button"
                        onClick={() => toggleSort(col.key)}
                        className={cx('inline-flex items-center gap-1 rounded hover:text-text-1', col.numeric && 'flex-row-reverse')}
                      >
                        {col.header}
                        {active &&
                          (sort.dir === 'asc' ? (
                            <ArrowUp size={12} strokeWidth={1.75} aria-hidden />
                          ) : (
                            <ArrowDown size={12} strokeWidth={1.75} aria-hidden />
                          ))}
                      </button>
                    ) : (
                      col.header
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
              const key = rowKey(row);
              const selected = key === selectedKey;
              return (
                <tr
                  key={key}
                  aria-selected={onSelect ? selected : undefined}
                  onClick={onSelect ? () => onSelect(row) : undefined}
                  className={cx(
                    'border-t border-border-subtle',
                    onSelect && 'cursor-pointer hover:bg-raised',
                    selected && 'bg-accent-weak shadow-[inset_2px_0_0_var(--color-accent)]',
                  )}
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={cx(
                        'px-3 text-text-1',
                        dense ? 'h-7' : 'h-9',
                        col.numeric ? 'num text-right' : 'text-left',
                      )}
                    >
                      {col.render(row)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {source && <p className="text-caption text-text-3">{source}</p>}
    </div>
  );
}
