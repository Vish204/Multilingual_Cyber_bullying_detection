import React from "react";
import SeverityChart from "./SeverityChart";
import PlatformChart from "./PlatformChart";

export default function DistributionRow({ severityData, platformData }) {
  return (
    <>
      {severityData && <SeverityChart severityData={severityData} />}
      {platformData && <PlatformChart platformData={platformData} />}
    </>
  );
}