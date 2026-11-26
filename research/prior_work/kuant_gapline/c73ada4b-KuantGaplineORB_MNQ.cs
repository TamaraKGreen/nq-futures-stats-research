// ============================================================================================
// KUANT GAPLINE ORB  --  MNQ Gap-Up Opening-Range Breakout, Long Only
// NinjaTrader 8 NinjaScript STRATEGY (C#)
// --------------------------------------------------------------------------------------------
// Instrument : MNQ (Micro E-mini Nasdaq-100), $2 per point
// Timeframe  : 1-minute chart; 5-minute signal bars are built inside the code
// Sessions   : CME US Index Futures ETH (17:00-16:00 CT). All rule times are Chicago time.
// Backtest   : May 2019 - Oct 2026, $500 risk/trade, costs included
//   Sharpe   : 2019-22 1.32 | 2023-24 1.91 | 2025-26 0.58
//              before 2024 1.35 | 2024 onward 1.22 | whole 1.30
//   Net $52,024 | max drawdown -$4,856 | 27 configurations tried
// --------------------------------------------------------------------------------------------
// HOW TO USE
//   - Chart: MNQ continuous (or the front month), 1-MINUTE bars, Trading Hours template
//     "CME US Index Futures ETH". Any display time zone works: the code converts every bar
//     to Chicago time itself. No second data series is needed.
//   - Calculate = On bar close. Entries are market orders that fill at the next 1-minute open.
//   - Commission ~$0.50 per contract per side; Strategy Analyzer slippage 1 tick.
//   - Load at least 15 trading days of history (the 200-EMA on 5-minute bars needs ~1,000
//     bars to settle; the gap needs one prior RTH day).
//   - ReEntryQty: 1 = re-enter the 17:00 leg with 1 MNQ (as backtested);
//     0 = re-enter with the same size as the position flattened at 15:40.
//   - Lower MaxContracts if you stack this with other strategies on one account.
//   - Do not run it alongside S27 on the same account (highly correlated).
//   - "Exit on session close" is ON (30 s). On normal days the position is already flat at 15:40;
//     it only acts on early-close holiday sessions (no 15:40 bar), which are then not re-entered.
//
// HOW IT EXECUTES
//   1. Gap: today's RTH open (08:30 CT 1-minute open) minus the prior RTH close (last bar
//      ending by 15:00 CT) must be >= +0.5 point.
//   2. Opening range = high/low of the 5-minute bars opening 08:30, 08:35, 08:40 CT.
//   3. EMA200 of all 5-minute closes, around the clock.
//   4. Long when a 5-minute bar opening 08:45 ... 15:30 CT closes above the range high AND
//      above the EMA200, while flat -> buy at the next 1-minute open. Repeat entries the same
//      day are allowed after an exit.
//   5. Stop distance = signal close - range low. Stop = fill - distance, target = fill + 1.5 x
//      distance. Size = floor($500 / (distance x $2)), 1..40 MNQ.
//   6. 15:40 CT: flat at market, no new entries until 17:00.
//   7. 17:00 CT: re-enter the flattened trade (1 MNQ) with the SAME stop and target prices.
//      The order is sent on the last bar of the session (15:59-16:00) and again on each bar
//      from 17:00 to 17:59 until it fills. Once filled, the old stop and target are worked; if
//      the reopen is already beyond either one, the trade is closed at market instead.
//
// CHECKED OUTSIDE NINJATRADER: run bar by bar through an NT8 order-API simulator on MNQ 1-minute
// data (May 2019 - Oct 2026) it reproduces the backtest: Sharpe 1.36 / 1.23 / 1.30 (before 2024 /
// 2024 on / whole), net $52,230 vs $52,024, daily P&L correlation 0.9999. Not yet compiled inside
// NinjaTrader itself: compile it, run it on a chart and compare a few trades before trading.
// Research code, not financial advice.
// ============================================================================================
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class KuantGaplineORB_MNQ : Strategy
    {
        private TimeZoneInfo dispZone, ctZone;
        private static readonly DateTime Epoch = new DateTime(1970, 1, 1);

        // RTH open / prior RTH close
        private DateTime rthDate = DateTime.MinValue;
        private double rthOpenPx, lastRthC, prevRthC;
        // internal 5-minute bars
        private long bKey = -1;
        private int bOpenMin;
        private DateTime bDate;
        private double bH, bL, bC, emaV;
        private bool emaOn, bDone = true;
        // opening range
        private DateTime rngDate = DateTime.MinValue;
        private double rngHi, rngLo;
        // position state
        private bool wasFlat = true, pendingNew;
        private string curSig = "";
        private double sdEntry, curStop, curTgt;
        // Lucid re-entry
        private int reDir, reQty;
        private double reStop, reTgt;
        private Order reStopOrd, reTgtOrd;
        // signal of the bar
        private bool sigL, sigS;
        private double sigC;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "KuantGaplineORB_MNQ";
                Description = "Kuant Gapline ORB - MNQ gap-up 15-minute ORB above EMA200 (5m), long only, Lucid 15:40 flat / 17:00 re-entry";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;     // only acts on early-close (holiday) sessions
                ExitOnSessionCloseSeconds = 30;
                BarsRequiredToTrade = 1;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                Slippage = 1;

                RngStart = 830; RngEnd = 845; LastSigTime = 1535; EmaLen = 200; TargetR = 1.5;
                LongOnly = true; UseGapFilter = true; GapMin = 0.5; RthStart = 830; RthEnd = 1500;
                RiskUSD = 500; MaxRiskUSD = 1000; MaxContracts = 40; FlatTime = 1540;
                ReenterAfterFlat = true; ReEntryQty = 1;
            }
            else if (State == State.Configure)
            {
                dispZone = TimeZoneInfo.Local;
                try { dispZone = NinjaTrader.Core.Globals.GeneralOptions.TimeZoneInfo ?? TimeZoneInfo.Local; } catch { dispZone = TimeZoneInfo.Local; }
                ctZone = TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time");
            }
        }

        // NT8 bar times are in the display time zone and stamped at the bar CLOSE.
        // Returns the bar's OPEN time in Chicago time (1-minute bars: close - 1 minute).
        private DateTime CtOpen(DateTime t)
        {
            DateTime utc = DateTime.SpecifyKind(t - dispZone.GetUtcOffset(DateTime.SpecifyKind(t, DateTimeKind.Unspecified)), DateTimeKind.Utc);
            return TimeZoneInfo.ConvertTimeFromUtc(utc, ctZone).AddMinutes(-1);
        }

        private static int Min(int hhmm) { return hhmm / 100 * 60 + hhmm % 100; }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || CurrentBar < 1) return;
            DateTime ct = CtOpen(Time[0]);
            int om = ct.Hour * 60 + ct.Minute;   // bar OPEN minute of day, CT
            int cm = om + 1;                     // bar CLOSE minute (MultiCharts-style time stamp)
            DateTime od = ct.Date;
            double pv = Instrument.MasterInstrument.PointValue;

            // ---------- RTH open and prior RTH close (1-minute bars closing 08:31 ... 15:00) ----------
            if (cm > Min(RthStart) && cm <= Min(RthEnd))
            {
                if (od != rthDate) { prevRthC = lastRthC; rthOpenPx = Open[0]; rthDate = od; }
                lastRthC = Close[0];
            }
            bool gapOK = rthDate == od && prevRthC > 0;
            double gapPts = gapOK ? rthOpenPx - prevRthC : 0;

            // ---------- internal 5-minute bars, EMA, opening range, signal ----------
            sigL = false; sigS = false;
            long key = ((long)(od - Epoch).TotalDays * 1440 + om) / 5;
            if (key != bKey)
            {
                if (!bDone && bKey >= 0) Close5(gapOK, gapPts);      // last minute of that 5m bar was missing
                bKey = key; bH = High[0]; bL = Low[0]; bOpenMin = om - om % 5; bDate = od; bDone = false;
            }
            bH = Math.Max(bH, High[0]); bL = Math.Min(bL, Low[0]); bC = Close[0];
            if (cm % 5 == 0) { Close5(gapOK, gapPts); bDone = true; }

            bool lucidBlock = cm >= Min(FlatTime) && cm < 1020;
            int d = Position.MarketPosition == MarketPosition.Long ? 1 : Position.MarketPosition == MarketPosition.Short ? -1 : 0;

            // ---------- first bar of a new position: remember its stop / target prices ----------
            if (d != 0 && wasFlat)
            {
                wasFlat = false;
                if (pendingNew)
                {
                    curStop = Position.AveragePrice - d * sdEntry;
                    curTgt = TargetR > 0 ? Position.AveragePrice + d * Math.Ceiling(TargetR * sdEntry / TickSize - 1e-9) * TickSize : 0;
                }
                pendingNew = false;
            }
            if (d == 0) wasFlat = true;

            // ---------- Lucid flat at 15:40 CT, remember the trade for the 17:00 re-entry ----------
            if (d != 0 && lucidBlock)
            {
                if (ReenterAfterFlat && reDir == 0)
                {
                    reDir = d; reStop = curStop; reTgt = curTgt;
                    reQty = ReEntryQty > 0 ? ReEntryQty : Position.Quantity;
                }
                CancelReExits();
                if (d == 1) ExitLong("GL_L_Flat", curSig); else ExitShort("GL_S_Flat", curSig);
                return;
            }

            // ---------- 17:00 re-entry: order on the 16:00 bar (fills at the 17:00 open); sent again on
            //            the 17:00-17:59 bars while still unfilled; dropped after 18:00 ----------
            if (reDir != 0 && d == 0)
            {
                if (cm == 960 || (cm >= 1021 && cm < 1080))
                {
                    // stop and target are sent from OnExecutionUpdate once the fill price is known
                    curSig = reDir == 1 ? "GL_LR" : "GL_SR";
                    curStop = reStop; curTgt = reTgt;
                    pendingNew = false;
                    if (reDir == 1) EnterLong(reQty, curSig); else EnterShort(reQty, curSig);
                }
                else if (cm >= 1080) reDir = 0;
                return;
            }

            // ---------- new entries (flat, outside 15:40-17:00) ----------
            if (d == 0 && reDir == 0 && !lucidBlock && (sigL || sigS))
            {
                double sd = sigL ? sigC - rngLo : rngHi - sigC;
                if (sd <= 0) return;
                int qty = (int)Math.Floor(RiskUSD / (sd * pv));
                qty = Math.Min(qty, (int)Math.Floor(MaxRiskUSD / (sd * pv)));
                qty = Math.Max(1, Math.Min(qty, MaxContracts));
                sdEntry = sd;
                curSig = sigL ? "GL_L" : "GL_S";
                SetStopLoss(curSig, CalculationMode.Ticks, Math.Max(1, Math.Round(sd / TickSize)), false);
                if (TargetR > 0) SetProfitTarget(curSig, CalculationMode.Ticks, Math.Max(1, Math.Ceiling(TargetR * sd / TickSize - 1e-9)));
                pendingNew = true;
                if (sigL) EnterLong(qty, curSig); else EnterShort(qty, curSig);
            }
        }


        // closes one internal 5-minute bar: EMA, opening range and the entry signal
        private void Close5(bool gapOK, double gapPts)
        {
            int t2 = bOpenMin + 5;               // 5-minute bar close minute (CT)
            if (!emaOn) { emaV = bC; emaOn = true; } else emaV += 2.0 / (EmaLen + 1) * (bC - emaV);
            if (t2 > Min(RngStart) && t2 <= Min(RngEnd))
            {
                if (bDate != rngDate) { rngHi = bH; rngLo = bL; rngDate = bDate; }
                else { rngHi = Math.Max(rngHi, bH); rngLo = Math.Min(rngLo, bL); }
            }
            bool rngReady = bDate == rngDate && t2 > Min(RngEnd);
            if (!rngReady || t2 > Min(LastSigTime)) return;
            sigC = bC;
            sigL = bC > rngHi && bC > emaV;
            sigS = !LongOnly && bC < rngLo && bC < emaV;
            if (UseGapFilter)
            {
                if (!gapOK) { sigL = false; sigS = false; }
                else { if (gapPts < GapMin) sigL = false; if (gapPts > -GapMin) sigS = false; }
            }
        }

        private void CancelReExits()
        {
            if (reStopOrd != null) { CancelOrder(reStopOrd); reStopOrd = null; }
            if (reTgtOrd != null) { CancelOrder(reTgtOrd); reTgtOrd = null; }
        }

        // 17:00 re-entry: once filled, work the old stop and target. If the reopen is already
        // through the stop or the target, close at market instead (a stop on the wrong side of
        // the market would be rejected).
        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            Order o = execution.Order;
            if (o == null || o.OrderState != OrderState.Filled) return;
            if (o.Name == "GL_LR" || o.Name == "GL_SR")
            {
                int d = o.Name == "GL_LR" ? 1 : -1;
                reDir = 0;
                double fill = o.AverageFillPrice;
                bool through = (fill - curStop) * d <= 0 || (curTgt > 0 && (fill - curTgt) * d >= 0);
                if (through) { if (d == 1) ExitLong("GL_LR_X", o.Name); else ExitShort("GL_SR_X", o.Name); return; }
                if (d == 1)
                {
                    reStopOrd = ExitLongStopMarket(0, true, o.Filled, curStop, "GL_LR_SL", o.Name);
                    if (curTgt > 0) reTgtOrd = ExitLongLimit(0, true, o.Filled, curTgt, "GL_LR_TP", o.Name);
                }
                else
                {
                    reStopOrd = ExitShortStopMarket(0, true, o.Filled, curStop, "GL_SR_SL", o.Name);
                    if (curTgt > 0) reTgtOrd = ExitShortLimit(0, true, o.Filled, curTgt, "GL_SR_TP", o.Name);
                }
            }
            else if (o.Name == "GL_LR_SL" || o.Name == "GL_SR_SL")
            {
                // stop filled: cancel the target
                reStopOrd = null; if (reTgtOrd != null) { CancelOrder(reTgtOrd); reTgtOrd = null; }
            }
            else if (o.Name == "GL_LR_TP" || o.Name == "GL_SR_TP")
            {
                // target filled: cancel the stop
                reTgtOrd = null; if (reStopOrd != null) { CancelOrder(reStopOrd); reStopOrd = null; }
            }
        }

        #region Properties
        [NinjaScriptProperty][Range(0, 2359)][Display(Name = "Range start (HHMM CT, 5m bar close >)", GroupName = "1. Opening range", Order = 1)]
        public int RngStart { get; set; }
        [NinjaScriptProperty][Range(0, 2359)][Display(Name = "Range end (HHMM CT, 5m bar close <=)", GroupName = "1. Opening range", Order = 2)]
        public int RngEnd { get; set; }
        [NinjaScriptProperty][Range(0, 2359)][Display(Name = "Last signal (HHMM CT, 5m bar close)", GroupName = "1. Opening range", Order = 3)]
        public int LastSigTime { get; set; }
        [NinjaScriptProperty][Range(1, 1000)][Display(Name = "EMA length (5m closes)", GroupName = "2. Rules", Order = 4)]
        public int EmaLen { get; set; }
        [NinjaScriptProperty][Range(0.0, 100.0)][Display(Name = "Target (x stop distance, 0 = none)", GroupName = "2. Rules", Order = 5)]
        public double TargetR { get; set; }
        [NinjaScriptProperty][Display(Name = "Long only", GroupName = "2. Rules", Order = 6)]
        public bool LongOnly { get; set; }
        [NinjaScriptProperty][Display(Name = "Use gap filter", GroupName = "2. Rules", Order = 7)]
        public bool UseGapFilter { get; set; }
        [NinjaScriptProperty][Range(0.0, 10000.0)][Display(Name = "Min gap (points)", GroupName = "2. Rules", Order = 8)]
        public double GapMin { get; set; }
        [NinjaScriptProperty][Range(0, 2359)][Display(Name = "RTH start (HHMM CT, 1m bar close >)", GroupName = "3. Session", Order = 9)]
        public int RthStart { get; set; }
        [NinjaScriptProperty][Range(0, 2359)][Display(Name = "RTH end (HHMM CT, 1m bar close <=)", GroupName = "3. Session", Order = 10)]
        public int RthEnd { get; set; }
        [NinjaScriptProperty][Range(0, 2359)][Display(Name = "Flat time (HHMM CT)", GroupName = "3. Session", Order = 11)]
        public int FlatTime { get; set; }
        [NinjaScriptProperty][Range(1.0, 100000.0)][Display(Name = "Risk per trade ($)", GroupName = "4. Sizing", Order = 12)]
        public double RiskUSD { get; set; }
        [NinjaScriptProperty][Range(1.0, 100000.0)][Display(Name = "Max risk per trade ($)", GroupName = "4. Sizing", Order = 13)]
        public double MaxRiskUSD { get; set; }
        [NinjaScriptProperty][Range(1, 1000)][Display(Name = "Max contracts", GroupName = "4. Sizing", Order = 14)]
        public int MaxContracts { get; set; }
        [NinjaScriptProperty][Display(Name = "Re-enter at 17:00 after the 15:40 flat", GroupName = "5. Lucid", Order = 15)]
        public bool ReenterAfterFlat { get; set; }
        [NinjaScriptProperty][Range(0, 1000)][Display(Name = "Re-entry contracts (0 = same as flattened)", GroupName = "5. Lucid", Order = 16)]
        public int ReEntryQty { get; set; }
        #endregion
    }
}
