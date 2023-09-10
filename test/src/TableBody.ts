export class TableBody {
    private body: HTMLTableSectionElement;

    constructor(table: HTMLTableElement) {
        this.body = table.createTBody();
    }

    public addEntry(entry: HTMLElement | string[]): HTMLTableRowElement {
        const row = this.body.insertRow();
        if (entry instanceof HTMLElement) {
            row.appendChild(entry);
        }
        else {
            for (let cell of entry) {
                const cellElement = row.insertCell();
                cellElement.textContent = cell;
            }
        }
        return row;
    }

    public updateEntry(entry: HTMLTableRowElement, entryData: Array<string | null>): boolean {
        if (entry.cells.length != entryData.length) {
            throw Error(`Number of entries in existing entry (${entry.cells.length}) don't match updated entry count (${updatedEntry.length})`);
        }

        for (let i = 0; i < entry.cells.length; i++) {
            if (entryData[i] === null) continue;

            const cell = entry.cells.item(i) as HTMLTableCellElement;
            cell.textContent = entryData[i];
        }
        
        return true;
    }
}