import { Device } from "../../../src";
import { NetworkGPT2 } from "./GPT2/GPT2.test";
import { NetworkMoonsData } from "./MoonsData/MoonsData.test";

function NetworksTest(device: Device) {
    NetworkMoonsData(device);
    NetworkGPT2(device);
}

export const NetworksTests = {category: "Networks", func: NetworksTest};