import { Device } from "../../../src";
import { NetworkMoonsData } from "./MoonsData/MoonsData.test";

function NetworksTest(device: Device) {
    NetworkMoonsData(device);
}

export const NetworksTests = {category: "Networks", func: NetworksTest};