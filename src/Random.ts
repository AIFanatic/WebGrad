export class Random {
    private static _seed = Math.floor(Math.random() * 100000000);
    
    public static SetRandomSeed(seed: number) {
        Random._seed = seed;
    }

    public static Random() {
        var t = Random._seed += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        t = ((t ^ t >>> 14) >>> 0) / 4294967296;
        
        Random._seed++;

        return t;
    }

    public static RandomRange(min: number, max: number): number {
        return Random.Random() * (max - min) + min;
    }
}