export class TableHeader {
    private head: HTMLTableSectionElement;

    constructor(table: HTMLTableElement) {
        this.head = table.createTHead();
    }

    public addEntry(entries: string[]) {
        const headerEntry = this.head.insertRow();

        for (let entry of entries) {
            const entryCell = headerEntry.insertCell();
            entryCell.textContent = entry;
        }
    }
}