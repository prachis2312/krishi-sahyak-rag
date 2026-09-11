"""Curated dataset of realistic Indian Agricultural and Farming Government Schemes.

Each scheme entry contains structured metadata, eligibility guidelines, financial benefits,
application procedures, and consolidated text chunks optimized for dense semantic embedding.
"""

from typing import List, Dict, Any

FARMING_SCHEMES: List[Dict[str, Any]] = [
    {
        "id": "pm-kisan",
        "name": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
        "short_name": "PM-KISAN",
        "category": "Direct Income Support",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Rs. 6,000 per year transferred directly into Aadhaar-linked bank accounts in three equal installments of Rs. 2,000 every 4 months (April-July, August-November, December-March).",
        "eligibility": "All landholding farmer families having cultivable land in their names. Exclusions: Institutional landholders, farmer families holding constitutional posts, serving/retired government employees, professionals (doctors, engineers, lawyers, CAs), and individuals who paid income tax in the last assessment year.",
        "documents_required": [
            "Aadhaar Card (mandatory)",
            "Land ownership documents / RoR (Record of Rights)",
            "Active Bank Account linked with Aadhaar and NPCI DBT",
            "Valid Mobile Number for e-KYC"
        ],
        "application_procedure": "Online self-registration via PM-KISAN portal (pmkisan.gov.in) under 'Farmers Corner' or through Common Service Centres (CSCs). Mandatory biometric or OTP-based e-KYC must be completed.",
        "key_highlights": "100% central sector scheme providing direct income support for agricultural inputs and domestic needs without any intermediary.",
        "full_text": (
            "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a Central Sector Scheme providing assured income support to all landholding farmer families across India. "
            "Under the scheme, financial benefit of Rs. 6,000 per annum is provided to eligible farmer families in three equal 4-monthly installments of Rs. 2,000 each. "
            "Funds are transferred directly into bank accounts through Direct Benefit Transfer (DBT). "
            "Eligibility: All landholder farmer families with cultivable landholding. "
            "Ineligible categories include institutional landholders, government employees, pensioners receiving Rs 10,000+ monthly, taxpayers, and registered professionals. "
            "Aadhaar linking and e-KYC completion are strictly mandatory for receiving installments."
        )
    },
    {
        "id": "pmfby",
        "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        "short_name": "PMFBY",
        "category": "Crop Insurance",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Comprehensive crop insurance coverage against non-preventable natural risks (drought, flood, pest, cyclone). Farmers pay an ultra-low uniform premium: 2.0% for Kharif food and oilseed crops, 1.5% for Rabi food and oilseed crops, and 5.0% for annual commercial/horticultural crops. The remaining actuarial premium is shared 50:50 by Central and State Governments.",
        "eligibility": "All farmers including sharecroppers and tenant farmers growing notified crops in notified areas are eligible. Voluntary for all farmers (both loanee and non-loanee).",
        "documents_required": [
            "Aadhaar Card",
            "Land Possession Certificate (LPC) / Land Revenue Receipt / Tenancy Agreement",
            "Sowing Certificate / Crop Sowing Declaration",
            "Bank Passbook copy with IFSC code"
        ],
        "application_procedure": "Enrollment through National Crop Insurance Portal (pmfby.gov.in), commercial banks, RRBs, Cooperative banks, or Common Service Centres (CSCs) before the cut-off dates for Kharif and Rabi seasons.",
        "key_highlights": "Covers standing crop losses, prevented sowing, mid-season adversity, post-harvest losses (up to 14 days for unbundled crops on field), and localized calamities (hailstorm, landslide, inundation). Claims settled using remote sensing, drone imagery, and CCE (Crop Cutting Experiments).",
        "full_text": (
            "Pradhan Mantri Fasal Bima Yojana (PMFBY) is India's flagship crop insurance scheme providing comprehensive financial protection against yield loss caused by non-preventable natural hazards. "
            "Farmers pay a nominal premium: 2% of sum insured for Kharif crops, 1.5% for Rabi crops, and 5% for annual commercial and horticultural crops. Balance premium subsidy is paid by government. "
            "Coverage stages: Prevented sowing/planting risk, standing crop yield losses (drought, dry spells, flood, pests), mid-season adversity, post-harvest losses up to 14 days, and localized calamities like hailstorms and cloudbursts. "
            "All farmers including sharecroppers and tenant farmers are eligible on voluntary basis."
        )
    },
    {
        "id": "kcc",
        "name": "Kisan Credit Card Scheme (KCC)",
        "short_name": "KCC",
        "category": "Agricultural Credit",
        "ministry": "Ministry of Finance & Ministry of Agriculture",
        "financial_assistance": "Short-term credit for cultivation expenses, post-harvest costs, and maintenance of farm assets. Concessional interest rate of 7% per annum for loans up to Rs. 3 Lakh. An additional 3% Prompt Repayment Incentive (PRI) reduces the effective interest rate to just 4% per annum. Collateral-free agricultural loans available up to Rs. 1.60 Lakh.",
        "eligibility": "Individual farmers, joint liability groups (JLGs), self-help groups (SHGs), tenant farmers, oral lessees, sharecroppers, as well as animal husbandry, dairy, and fisheries farmers.",
        "documents_required": [
            "Duly filled KCC Application Form",
            "Identity and Address Proof (Aadhaar, Voter ID, PAN)",
            "Land ownership records (Khasra/Khatauni) or tenancy agreement",
            "Passport-sized photographs"
        ],
        "application_procedure": "Apply at any Commercial Bank, Regional Rural Bank (RRB), Primary Agricultural Credit Society (PACS), or apply online through bank portals and Jan Samarth portal.",
        "key_highlights": "Revolving cash credit account valid for 5 years with annual review; includes ATM-cum-debit card (RuPay Kisan Card) for hassle-free withdrawals and input purchases.",
        "full_text": (
            "The Kisan Credit Card (KCC) scheme provides farmers with timely, flexible, and affordable short-term institutional credit. "
            "Credit limit covers crop production expenses, post-harvest expenses, consumption requirements of farmer households, and farm asset maintenance. "
            "Interest subvention: Base interest is 7% for crop loans up to Rs. 3,00,000. With 3% prompt repayment incentive, the effective interest rate is 4% per annum. "
            "Collateral is waived for loans up to Rs. 1.60 lakh (extendable to Rs 2 lakh for tie-up arrangements). "
            "Now expanded to cover Dairy, Animal Husbandry, and Fisheries working capital needs up to Rs. 2 lakh limit within overall ceiling."
        )
    },
    {
        "id": "pmksy-pdmc",
        "name": "Pradhan Mantri Krishi Sinchayee Yojana - Per Drop More Crop (PMKSY-PDMC)",
        "short_name": "PMKSY (Per Drop More Crop)",
        "category": "Micro Irrigation",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Financial subsidy for installation of micro-irrigation systems (Drip and Sprinkler irrigation). Small and Marginal farmers receive up to 55% financial assistance of benchmark cost; other farmers receive up to 45% financial assistance.",
        "eligibility": "All categories of farmers having cultivable land with an assured water source (borewell, open well, farm pond, canal). Group of farmers, cooperative societies, and water user associations are also eligible.",
        "documents_required": [
            "Aadhaar Card",
            "Land ownership records (7/12 extract, Khatian, ROR)",
            "Water and power source availability certificate",
            "Soil and water test report",
            "Bank passbook copy"
        ],
        "application_procedure": "Application through State Department of Agriculture / Horticulture portals or District Horticulture Officer (DHO) office with quotation from empanelled micro-irrigation manufacturers.",
        "key_highlights": "Increases water use efficiency by 40-50%, saves electricity/fertilizers (fertigation), improves crop yield by 20-30%, and prevents soil erosion.",
        "full_text": (
            "Pradhan Mantri Krishi Sinchayee Yojana - Per Drop More Crop (PMKSY-PDMC) focuses on enhancing water use efficiency at farm level through micro-irrigation technologies. "
            "The scheme provides substantial financial assistance: 55% subsidy for Small and Marginal farmers (holding under 2 hectares), and 45% subsidy for other category farmers for installing Drip Irrigation and Sprinkler systems. "
            "Promotes precision water management, fertigation (applying fertilizers through irrigation), and reduces input cost while increasing productivity. "
            "Eligible beneficiaries: All farmers having cultivable land with accessible water sources."
        )
    },
    {
        "id": "soil-health-card",
        "name": "Soil Health Card Scheme (SHC)",
        "short_name": "Soil Health Card",
        "category": "Soil Fertility & Nutrient Management",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Free soil testing and customized Soil Health Card issued every 2 years to farmers. Cards provide status on 12 chemical parameters: Macro-nutrients (N, P, K), Secondary-nutrient (S), Micro-nutrients (Zn, Fe, Cu, Mn, Bo), and Physical parameters (pH, EC, Organic Carbon).",
        "eligibility": "All farmers across all States and Union Territories with agricultural land.",
        "documents_required": [
            "Farmer basic details (Name, Village, Contact)",
            "Aadhaar Number",
            "Khasra/Khatauni land identification number for soil sample geolocation"
        ],
        "application_procedure": "Soil samples collected by local agriculture extension officers, Krishi Vigyan Kendras (KVKs), or Village Level Entrepreneurs (VLEs). Farmers can view and download their digital Soil Health Card from soilhealth.dac.gov.in.",
        "key_highlights": "Recommends balanced, crop-specific dosage of chemical fertilizers and bio-fertilizers; prevents overuse of urea, reduces input costs by 15-20%, and sustains soil fertility.",
        "full_text": (
            "The Soil Health Card (SHC) Scheme promotes integrated nutrient management by issuing soil health status reports to every landholder farmer on a 2-year cycle. "
            "Tests 12 critical soil parameters: Nitrogen (N), Phosphorus (P), Potassium (K), Sulphur (S), Zinc (Zn), Iron (Fe), Copper (Cu), Manganese (Mn), Boron (Bo), pH, Electrical Conductivity (EC), and Organic Carbon (OC). "
            "Provides customized, crop-wise fertilizer recommendations including organic manure and bio-fertilizers to prevent soil degradation, optimize fertilizer expenses, and enhance farm yield."
        )
    },
    {
        "id": "enam",
        "name": "National Agriculture Market (e-NAM)",
        "short_name": "e-NAM",
        "category": "Agricultural Marketing",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Zero platform fee for farmers. 100% centrally funded integration of wholesale APMC mandis. Free quality assaying of produce and direct online e-payment to farmers' bank accounts upon auction completion.",
        "eligibility": "All farmers, Farmer Producer Organisations (FPOs), commission agents, and licensed traders.",
        "documents_required": [
            "Aadhaar Card",
            "Bank Account details (passbook / cancelled cheque)",
            "Mobile number for SMS transaction alerts"
        ],
        "application_procedure": "Register online at enam.gov.in or at the e-NAM facilitation desk at any integrated APMC mandi. Produce is weighed, quality-tested, and put up for national electronic bidding.",
        "key_highlights": "Unites over 1,300+ APMC mandis across India into a single digital marketplace; removes geographical barriers, eliminates middleman cartels, and ensures transparent price discovery.",
        "full_text": (
            "National Agriculture Market (e-NAM) is a pan-India electronic trading portal that networks existing APMC mandis to create a unified national market for agricultural commodities. "
            "Key features: Transparent online bidding process, scientific quality assaying facilities within mandis, electronic warehouse receipt trading, and direct online payment to farmer accounts. "
            "Enables farmers to access buyers and traders across state boundaries, securing better prices for their harvested produce without intermediary commission cuts."
        )
    },
    {
        "id": "smam",
        "name": "Sub-Mission on Agricultural Mechanization (SMAM)",
        "short_name": "SMAM",
        "category": "Farm Mechanization",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "40% to 50% capital subsidy on procurement of agricultural machinery and equipment (tractors, power tillers, rotavators, multi-crop threshers, combine harvesters, seed drills). For establishing Custom Hiring Centres (CHCs), financial assistance of up to 40% (max Rs. 10 Lakhs to Rs. 24 Lakhs depending on project cost) is provided.",
        "eligibility": "Individual farmers (with higher subsidy percentage for Small, Marginal, SC, ST, and Women farmers), Farmer Producer Organisations (FPOs), Cooperative Societies, and Rural Entrepreneurs.",
        "documents_required": [
            "Aadhaar Card",
            "Land records (7/12 extract, Jamabandi)",
            "Bank passbook",
            "Caste certificate (if applicable for enhanced SC/ST subsidy)",
            "Quotation from authorized machinery dealer"
        ],
        "application_procedure": "Online application on Central SMAM portal (agrimachinery.nic.in) or State Agriculture Engineering Directorate portals through Direct Benefit Transfer in Agriculture Mechanization (DBT-AM).",
        "key_highlights": "Overcomes farm labour shortage, enables timely agricultural operations, and makes modern farm machinery accessible to smallholders via Custom Hiring Centres.",
        "full_text": (
            "Sub-Mission on Agricultural Mechanization (SMAM) aims to increase the reach of farm mechanization to small and marginal farmers and regions with low farm power availability. "
            "Financial assistance: 40% to 50% subsidy on individual machinery purchase like power tillers, laser land levellers, zero-till seed drills, and power reapers. "
            "Provides up to 40% project cost assistance for establishing village-level Custom Hiring Centres (CHCs) and High-Tech Machinery Hubs, allowing small farmers to hire advanced machinery on nominal rental basis."
        )
    },
    {
        "id": "aif",
        "name": "Agriculture Infrastructure Fund (AIF)",
        "short_name": "AIF",
        "category": "Post-Harvest Infrastructure",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Medium to long-term debt financing facility for investment in viable post-harvest management infrastructure and community farming assets. Provides 3% per annum interest subvention on loans up to Rs. 2 Crore for a maximum tenure of 7 years. Credit guarantee coverage under CGTMSE for loans up to Rs. 2 Crore with fee paid by Government.",
        "eligibility": "Primary Agricultural Credit Societies (PACS), Marketing Cooperative Societies, FPOs, Self Help Groups (SHGs), Farmers, Joint Liability Groups, Agri-entrepreneurs, and Startups.",
        "documents_required": [
            "Detailed Project Report (DPR)",
            "KYC documents (Aadhaar, PAN)",
            "Land ownership / lease agreement for infrastructure site",
            "Bank loan application & audited financials (if applicable)"
        ],
        "application_procedure": "Register and submit project proposal through the dedicated AIF portal (agriinfra.dac.gov.in). Evaluated and sanctioned by participating commercial and cooperative banks.",
        "key_highlights": "Covers cold stores, modern silos, supply chain logistics, ripening chambers, sorting/grading units, pack-houses, e-marketing hubs, and bio-stimulant production units.",
        "full_text": (
            "The Agriculture Infrastructure Fund (AIF) is a financing facility created to mobilize medium-to-long term debt for post-harvest management infrastructure and community farming assets. "
            "Benefits: 3% interest subvention per annum for loans up to Rs. 2 Crore for up to 7 years, along with credit guarantee coverage under CGTMSE. "
            "Eligible projects include cold chain networks, assaying and sorting units, primary processing centres, smart warehouses, pack-houses, and solar-powered cold storages. "
            "Eligible borrowers: Farmers, FPOs, Agri-startups, PACS, and Agri-entrepreneurs."
        )
    },
    {
        "id": "pkvy",
        "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
        "short_name": "PKVY",
        "category": "Organic Farming",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Financial assistance of Rs. 50,000 per hectare for a 3-year period. Out of this, Rs. 31,000 per hectare (62%) is provided directly to farmers through DBT for purchasing organic inputs (organic fertilizers, bio-pesticides, high-grade seeds, vermicompost), and Rs. 8,800/ha for post-harvest, packaging, branding, and marketing.",
        "eligibility": "Farmers forming organic farming clusters of 20 hectares (or 50 acres) in contiguous areas. Individual farmers within the cluster are eligible.",
        "documents_required": [
            "Aadhaar Card",
            "Land possession records",
            "Bank passbook details",
            "Cluster membership endorsement"
        ],
        "application_procedure": "Apply through Regional Council under Participatory Guarantee System of India (PGS-India) or State Department of Agriculture / Organic Farming Nodal Agency.",
        "key_highlights": "Promotes chemical-free organic farming, PGS-India green and organic certification with zero certification charges for farmers, and establishes premium domestic marketing channels.",
        "full_text": (
            "Paramparagat Krishi Vikas Yojana (PKVY) is an initiative to promote organic farming through a cluster approach and Participatory Guarantee System (PGS) certification. "
            "Under PKVY, financial assistance of Rs. 50,000 per hectare is provided over 3 years. "
            "Direct Benefit: Rs. 31,000/ha is credited directly to farmers for organic inputs (bio-fertilizers, vermicompost, indigenous botanical extracts). "
            "Remaining funds support soil sampling, cluster formation, PGS-India certification, value addition, packaging, and branding organic produce for domestic and export markets."
        )
    },
    {
        "id": "pm-kmy",
        "name": "Pradhan Mantri Kisan Maandhan Yojana (PM-KMY)",
        "short_name": "PM-KMY",
        "category": "Old Age Pension & Social Security",
        "ministry": "Ministry of Agriculture and Farmers Welfare (Administered by LIC)",
        "financial_assistance": "Assured monthly pension of Rs. 3,000 to eligible small and marginal farmers upon reaching the age of 60 years. In case of beneficiary's demise, spouse receives 50% family pension (Rs. 1,500/month).",
        "eligibility": "Small and Marginal Farmers (SMFs) owning cultivable land up to 2 hectares (5 acres), aged between 18 to 40 years. Monthly voluntary contribution ranges between Rs. 55 to Rs. 200 depending on entry age; equal matching 50% contribution is deposited by Central Government.",
        "documents_required": [
            "Aadhaar Card",
            "Savings Bank Account / PMJDY Account passbook with auto-debit consent",
            "Land ownership record (Khata/Khatoni)"
        ],
        "application_procedure": "Enrollment through nearest Common Service Centre (CSC) or online self-enrollment on maandhan.in. Voluntary contribution can also be auto-debited directly from PM-KISAN installment credits.",
        "key_highlights": "Voluntary and contributory pension scheme providing social security and financial independence in old age for smallholders.",
        "full_text": (
            "Pradhan Mantri Kisan Maandhan Yojana (PM-KMY) is an old-age social security pension scheme for Small and Marginal Farmers (cultivable landholding up to 2 hectares). "
            "Provides an assured monthly pension of Rs. 3,000 after attaining 60 years of age. "
            "Entry age: 18 to 40 years. Beneficiaries make monthly contributions between Rs. 55 and Rs. 200, matched 100% by the Central Government. "
            "Administered by Life Insurance Corporation of India (LIC). Farmers enrolled in PM-KISAN can opt to have their monthly contribution directly deducted from PM-KISAN quarterly payouts."
        )
    },
    {
        "id": "midh",
        "name": "Mission for Integrated Development of Horticulture (MIDH)",
        "short_name": "MIDH",
        "category": "Horticulture Development",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Subsidies ranging from 35% to 50% for establishment of new orchards, high-density plantations, hi-tech polyhouses, shade net houses, mushrooms units, protected vegetable cultivation, and post-harvest pack-houses.",
        "eligibility": "Individual farmers, groups of farmers, FPOs, registered societies, and agri-entrepreneurs engaging in horticulture crops (fruits, vegetables, root/tuber crops, mushrooms, spices, flowers, aromatic plants).",
        "documents_required": [
            "Aadhaar Card",
            "Land ownership / long lease documents",
            "Bank passbook with IFSC code",
            "Technical project proposal with cost estimates"
        ],
        "application_procedure": "Apply to District Horticulture Officer (DHO) / State Horticulture Mission (SHM) portal.",
        "key_highlights": "Covers end-to-end horticulture value chain from planting material, protected greenhouse farming, drip fertigation, to cold storage and marketing infrastructure.",
        "full_text": (
            "Mission for Integrated Development of Horticulture (MIDH) is a centrally sponsored scheme for the holistic growth of the horticulture sector covering fruits, vegetables, root and tuber crops, mushrooms, spices, flowers, and bamboo. "
            "Offers 35% to 50% capital subsidy for establishing high-tech nurseries, protected cultivation structures (polyhouses, shade net houses), plastic mulching, mushroom spawn production, and on-farm pack houses. "
            "Aims to double horticulture productivity, promote crop diversification away from water-intensive cereals, and enhance farmers' cash income."
        )
    },
    {
        "id": "pmmsy",
        "name": "Pradhan Mantri Matsya Sampada Yojana (PMMSY)",
        "short_name": "PMMSY",
        "category": "Fisheries & Aquaculture",
        "ministry": "Ministry of Fisheries, Animal Husbandry and Dairying",
        "financial_assistance": "Financial assistance of 40% of unit cost for general category beneficiaries and up to 60% for SC, ST, and Women beneficiaries for establishment of inland aquaculture, Biofloc units, Recirculatory Aquaculture Systems (RAS), fish seed hatcheries, insulated trucks, and motorized fishing boats.",
        "eligibility": "Fishers, fish farmers, fish workers, fish vendors, SHGs, Joint Liability Groups, Fisheries Cooperatives, and Aquaculture Entrepreneurs.",
        "documents_required": [
            "Aadhaar Card",
            "Land / water body lease or ownership documents",
            "Bank passbook",
            "Detailed Project Report (DPR)"
        ],
        "application_procedure": "Submit application on PMMSY portal (pmmsy.dof.gov.in) or through District Fisheries Office (DFO).",
        "key_highlights": "Aims to boost national fish production, modernize cold chain infrastructure, generate 55+ lakh livelihood opportunities, and double fishers' income.",
        "full_text": (
            "Pradhan Mantri Matsya Sampada Yojana (PMMSY) is a flagship scheme to drive the Blue Revolution in India by developing sustainable fisheries and aquaculture. "
            "Provides financial assistance: 40% project subsidy for general category, and 60% subsidy for SC/ST/Women beneficiaries. "
            "Covers construction of new ponds, high-density Biofloc units, Recirculatory Aquaculture Systems (RAS), ornamental fish breeding units, deep-sea fishing vessels, solar-powered fish drying units, and three-wheeler/four-wheeler refrigerated fish vending vehicles."
        )
    },
    {
        "id": "rkvy-raftaar",
        "name": "Rashtriya Krishi Vikas Yojana - RAFTAAR (Agri-Startup Grants)",
        "short_name": "RKVY-RAFTAAR",
        "category": "Agri-Entrepreneurship & Startups",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Grant-in-aid of up to Rs. 5 Lakhs for idea-stage agripreneurs (Agripreneurship Orientation Programme) with 2-month stipend (Rs 10,000/month), and seed-stage grant of up to Rs. 25 Lakhs (85% grant, 15% incubatee share) for registered early-stage agricultural startups.",
        "eligibility": "Students, rural youth, farmers, researchers, and innovators with innovative ideas or prototypes in agriculture and allied domains (IoT in farming, bio-fertilizers, farm robotics, supply chain tech).",
        "documents_required": [
            "PAN Card & Aadhaar Card",
            "Company incorporation certificate / DPIIT recognition (for seed grant)",
            "Business pitch deck and prototype demonstration",
            "Bank details"
        ],
        "application_procedure": "Apply to designated Knowledge Partners (KPs) and RKVY-RAFTAAR Agribusiness Incubators (R-ABIs) such as IARI, MANAGE, NIAM, and agricultural universities.",
        "key_highlights": "Fosters agricultural entrepreneurship, tech innovation, supply chain modernization, and rural wealth creation through non-dilutive grant funding.",
        "full_text": (
            "RKVY-RAFTAAR (Rashtriya Krishi Vikas Yojana - Remunerative Approaches for Agriculture and Allied Sector Rejuvenation) promotes agribusiness and entrepreneurship by providing financial mentoring and incubation grants. "
            "Financial Grant: Up to Rs. 5 Lakhs grant for idea-stage innovators during 2-month incubation with monthly stipend. "
            "Up to Rs. 25 Lakhs grant-in-aid for seed-stage startups holding minimum viable products (MVP) in farm tech, post-harvest solutions, organic farming inputs, and precision agriculture."
        )
    },
    {
        "id": "kisan-drone",
        "name": "Kisan Drone Scheme & Subsidy",
        "short_name": "Kisan Drone Scheme",
        "category": "Agri-Tech & Precision Farming",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "financial_assistance": "Up to 100% cost of drone (max Rs. 10 Lakhs) for ICAR institutes, KVKs, and State Agriculture Universities for demonstrations. Up to 75% subsidy for FPOs. Financial assistance of 50% (max Rs. 5 Lakhs) for SC/ST, Small/Marginal, and Women farmers; and 40% (max Rs. 4 Lakhs) for other individual farmers purchasing agricultural drones.",
        "eligibility": "Individual farmers, FPOs, Custom Hiring Centres, Cooperative Societies, Rural Entrepreneurs, and Agricultural Institutions.",
        "documents_required": [
            "Aadhaar Card",
            "Land ownership documents",
            "DGCA-approved Remote Pilot Certificate / Training Certificate",
            "Bank passbook copy"
        ],
        "application_procedure": "Apply through SMAM / DBT Agriculture portal (agrimachinery.nic.in) or State Agriculture Engineering Department.",
        "key_highlights": "Enables ultra-fast, precision spraying of liquid fertilizers (nano-urea, nano-DAP) and pesticides over 1 acre in under 7 minutes, saving 90% water and reducing farmer chemical exposure.",
        "full_text": (
            "The Kisan Drone Scheme promotes the adoption of unmanned aerial vehicles (drones) in agriculture for precision crop monitoring, soil health assessment, and aerial spraying of nano-fertilizers and bio-pesticides. "
            "Subsidy structure: Up to 100% (max Rs. 10 Lakh) grant for ICAR/SAUs/KVKs for field demonstrations; up to 75% for FPOs; 50% (max Rs. 5 Lakh) subsidy for Small/Marginal/SC/ST/Women farmers; and 40% (max Rs. 4 Lakh) for general category farmers. "
            "Reduces spraying time from hours to 7-10 minutes per acre, protects farmers from hazardous chemical exposure, and ensures uniform droplet distribution."
        )
    }
]
