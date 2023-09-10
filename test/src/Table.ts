import { TableBody } from "./TableBody";
import { TableHeader } from "./TableHeader";

export class Table {
    private container: HTMLDivElement;
    private table: HTMLTableElement;
    public tableHead: TableHeader;
    public tableBody: TableBody;

    constructor(container: HTMLDivElement) {
        this.container = container;
        this.table = document.createElement("table");
        this.tableHead = new TableHeader(this.table);
        this.tableBody = new TableBody(this.table);

        this.container.appendChild(this.table);
    }
}